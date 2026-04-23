import motmetrics as mm
import pandas as pd
import numpy as np
import warnings
from prettytable import PrettyTable
warnings.filterwarnings('ignore', message='The behavior of DataFrame concatenation with empty or all-NA entries is deprecated')


def read_file(path, id_field):
    if path.endswith('csv'):
        return read_csv(path)
    elif path.endswith('xlsx'):
        return read_excel(path, id_field)
    elif path.endswith('txt'):
        return read_txt(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def read_txt(txt_path):
    df = pd.read_csv(txt_path,
                     sep=',',
                     names=['frame_number',
                            'identity_id',
                            'left',
                            'top',
                            'width',
                            'height',
                            'score',
                            'x',
                            'y',
                            'z'])
    return df


def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df


def read_excel(excel_path, id_field):
    all_dfs = []
    xls = pd.ExcelFile(excel_path)
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        all_dfs.append(df)
    result_df = pd.concat(all_dfs, ignore_index=True)
    return result_df


class TrackEval:
    def __init__(self, if_kuajing=False):
        self.record_mem = {}
        self.global_count = 1
        self.if_kuajing = if_kuajing
        self.acc = mm.MOTAccumulator(auto_id=True)

    def _process_detection_csv(self, df, id_field='primary_uuid', mark_id=0):
        mem = {}
        for index, row in df.iterrows():
            idx_frame = row["idx_frame"]
            box_x1 = row["box_x1"]
            box_y1 = row["box_y1"]
            box_w = row["box_x2"] - row["box_x1"]
            box_h = row["box_y2"] - row["box_y1"]
            primary_uuid = row[id_field]
            camera_id = row["camera_id"]
            camera_name = row["camera_name"]

            if primary_uuid not in self.record_mem:
                self.record_mem[primary_uuid] = self.global_count
                self.global_count += 1
            if mark_id == 0:
                mark = str(camera_id) + "_" + str(camera_name) + "_" + str(idx_frame)
            elif mark_id == 1:
                mark = str(int(idx_frame))
            else:
                raise ValueError()
            if mark not in mem:
                mem[mark] = []
            mem[mark].append([box_x1, box_y1, box_w, box_h, self.record_mem[primary_uuid]])
        return mem

    def _process_detection_txt(self, df, mark_id=0):
        mem = {}
        for index, row in df.iterrows():
            idx_frame = int(row['frame_number'])
            box_x1 = row['left']
            box_y1 = row['top']
            box_w = row['width']
            box_h = row['height']
            primary_uuid = row['identity_id']

            if primary_uuid not in self.record_mem:
                self.record_mem[primary_uuid] = self.global_count
                self.global_count += 1
            if mark_id == 0:
                mark = str(idx_frame)
            elif mark_id == 1:
                mark = str(idx_frame)
            else:
                raise ValueError()
            if mark not in mem:
                mem[mark] = []
            mem[mark].append([box_x1, box_y1, box_w, box_h, self.record_mem[primary_uuid]])
        return mem

    def _process_detection(self, df, suffix, id_field, mark):
        if suffix == 'txt':
            return self._process_detection_txt(df, mark)
        elif suffix == 'csv' or suffix == 'xlsx':
            return self._process_detection_csv(df, id_field, mark)

    def evaluate(self, pr_path, gt_path, id_field, mark_id):
        pr_df = read_file(pr_path, id_field)
        gt_df = read_file(gt_path, id_field)
        pr_suffix = pr_path.split('.')[-1]
        gt_suffix = gt_path.split('.')[-1]

        pr_id = 'oid'
        gt_id = 'oid'

        if self.if_kuajing:
            pr_id = 'secondary_uuid'

        mem_pr = self._process_detection(pr_df, pr_suffix, pr_id, 0)
        mem_gt = self._process_detection(gt_df, gt_suffix, gt_id, 1)

        if self.if_kuajing:
            for k, v in mem_gt.items():
                for value in v:
                    value[-1] = 1

        total_predictions = 0
        total_ground_truth = 0

        matched_frames = 0
        for k in mem_gt:
            pr_key = None
            for pr_k in mem_pr.keys():
                if pr_k.endswith(f"_{k}"):
                    pr_key = pr_k
                    break

            if pr_key and pr_key in mem_pr:
                matched_frames += 1

                gtx1y1wh = np.array(mem_gt[k])[:, :4]
                prx1y1wh = np.array(mem_pr[pr_key])[:, :4]

                gt_id = np.array(mem_gt[k])[:, -1].tolist()
                pr_id = np.array(mem_pr[pr_key])[:, -1].tolist()

                total_predictions += len(pr_id)
                total_ground_truth += len(gt_id)

                assert len(gt_id) == gtx1y1wh.shape[0]
                assert len(pr_id) == prx1y1wh.shape[0]
                self.acc.update(
                    gt_id,
                    pr_id,
                    mm.distances.iou_matrix(
                        gtx1y1wh,
                        prx1y1wh,
                        max_iou=0.5
                    )
                )

    def get_result(self):
        mh = mm.metrics.create()
        summary = mh.compute(
            self.acc,
            metrics=mm.metrics.motchallenge_metrics,
            name='acc'
        )

        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )

        IDF1 = np.round(summary['idf1'].values[0] * 100, 2)
        MOTA = np.round(summary['mota'].values[0] * 100, 2)
        IDs = summary['num_switches'].values[0]
        GTs = summary['num_unique_objects'].values[0]

        table = PrettyTable()
        table.field_names = ["IDF1(up)", "MOTA(up)", "IDs(down)", "GTs"]
        table.add_row([IDF1, MOTA, IDs, GTs])

        print(table)
        print(strsummary)

        return {
            'IDF1': IDF1,
            'MOTA': MOTA,
            'IDs': IDs,
            'GTs': GTs
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="Prediction results CSV file")
    parser.add_argument("--gt_file", type=str, required=True, help="Ground truth file")
    parser.add_argument("--id_field", type=str, default='oid', help="ID field name")
    args = parser.parse_args()

    evaluator = TrackEval()
    evaluator.evaluate(args.pred_file, args.gt_file, id_field=args.id_field, mark_id=1)
    evaluator.get_result()

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


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    return inter_area / (union_area + 1e-6)


def linear_assignment(cost_matrix, threshold=0.5):
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    n_rows, n_cols = cost_matrix.shape
    rows = list(range(n_rows))
    cols = list(range(n_cols))
    matches = []
    unmatched_rows = []
    unmatched_cols = []

    while rows and cols:
        min_val = float('inf')
        min_i, min_j = -1, -1

        for i in rows:
            for j in cols:
                if cost_matrix[i, j] < min_val and cost_matrix[i, j] < threshold:
                    min_val = cost_matrix[i, j]
                    min_i, min_j = i, j

        if min_i >= 0:
            matches.append((min_i, min_j))
            rows.remove(min_i)
            cols.remove(min_j)
        else:
            break

    unmatched_rows = rows
    unmatched_cols = cols

    return matches, unmatched_rows, unmatched_cols


class TrackEval:
    def __init__(self, if_kuajing=False):
        self.record_mem = {}
        self.global_count = 1
        self.if_kuajing = if_kuajing
        
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.id_switches = 0
        self.gt_ids = set()
        self.pr_ids = set()
        self.frame_matches = {}
        self.track_assignments = {}

    def _process_detection_csv(self, df, id_field='primary_uuid', mark_id=0):
        mem = {}
        for index, row in df.iterrows():
            idx_frame = row["idx_frame"]
            box_x1 = row["box_x1"]
            box_y1 = row["box_y1"]
            box_w = row["box_x2"] - row["box_x1"]
            box_h = row["box_y2"] - row["box_y1"]
            primary_uuid = row[id_field]
            camera_id = row.get("camera_id", "1")
            camera_name = row.get("camera_name", "scene1")

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
            mark = str(idx_frame)
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

        pr_id = id_field
        gt_id = id_field

        if self.if_kuajing:
            pr_id = 'secondary_uuid'

        mem_pr = self._process_detection(pr_df, pr_suffix, pr_id, 0)
        mem_gt = self._process_detection(gt_df, gt_suffix, gt_id, 1)

        for k in mem_gt:
            for value in mem_gt[k]:
                self.gt_ids.add(value[-1])

        for k in mem_pr:
            for value in mem_pr[k]:
                self.pr_ids.add(value[-1])

        for frame in mem_gt:
            pr_key = None
            for pk in mem_pr.keys():
                if pk.endswith(f"_{frame}"):
                    pr_key = pk
                    break

            if pr_key and pr_key in mem_pr:
                gt_boxes = np.array(mem_gt[frame])[:, :4]
                pr_boxes = np.array(mem_pr[pr_key])[:, :4]
                gt_ids = np.array(mem_gt[frame])[:, -1].tolist()
                pr_ids = np.array(mem_pr[pr_key])[:, -1].tolist()

                n_gt = len(gt_boxes)
                n_pr = len(pr_boxes)

                cost_matrix = np.ones((n_gt, n_pr)) * 2.0

                for i in range(n_gt):
                    for j in range(n_pr):
                        cost_matrix[i, j] = 1 - iou(gt_boxes[i], pr_boxes[j])

                matches, unmatched_gt, unmatched_pr = linear_assignment(cost_matrix, threshold=0.5)

                self.tp += len(matches)
                self.fn += len(unmatched_gt)
                self.fp += len(unmatched_pr)

                self.frame_matches[frame] = []
                for i, j in matches:
                    self.frame_matches[frame].append((gt_ids[i], pr_ids[j]))

                    if gt_ids[i] in self.track_assignments:
                        if self.track_assignments[gt_ids[i]] != pr_ids[j]:
                            self.id_switches += 1
                    self.track_assignments[gt_ids[i]] = pr_ids[j]

    def get_result(self):
        total_objects = len(self.gt_ids)
        total_predictions = len(self.pr_ids)

        IDP = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        IDR = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        IDF1 = 2 * IDP * IDR / (IDP + IDR) if (IDP + IDR) > 0 else 0

        table = PrettyTable()
        table.field_names = ["IDF1(up)", "IDP(up)", "IDR(up)", "IDs(down)", "GTs"]
        table.add_row([round(IDF1*100, 2), round(IDP*100, 2), round(IDR*100, 2), self.id_switches, total_objects])

        print(table)
        print("\nDetailed Results:")
        print(f"True Positives: {self.tp}")
        print(f"False Positives: {self.fp}")
        print(f"False Negatives: {self.fn}")

        return {
            'IDF1': round(IDF1*100, 2),
            'IDP': round(IDP*100, 2),
            'IDR': round(IDR*100, 2),
            'IDs': self.id_switches,
            'GTs': total_objects
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="Prediction results CSV file")
    parser.add_argument("--gt_file", type=str, required=True, help="Ground truth file")
    parser.add_argument("--id_field", type=str, default='primary_uuid', help="ID field name")
    args = parser.parse_args()

    evaluator = TrackEval()
    evaluator.evaluate(args.pred_file, args.gt_file, id_field=args.id_field, mark_id=1)
    evaluator.get_result()
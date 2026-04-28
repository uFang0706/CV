import pandas as pd
import numpy as np
import warnings
from prettytable import PrettyTable
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings('ignore', message='The behavior of DataFrame concatenation with empty or all-NA entries is deprecated')


def read_file(path, id_field=None):
    """Read tracking/ground-truth files in CSV, XLSX or MOT txt format."""
    if path.endswith('csv'):
        return pd.read_csv(path)
    if path.endswith('xlsx'):
        xls = pd.ExcelFile(path)
        return pd.concat([xls.parse(sheet) for sheet in xls.sheet_names], ignore_index=True)
    if path.endswith('txt'):
        return pd.read_csv(
            path,
            sep=',',
            names=['frame_number', 'identity_id', 'left', 'top', 'width', 'height', 'score', 'x', 'y', 'z']
        )
    raise ValueError(f"Unsupported file format: {path}")


def iou(box1, box2):
    """IoU for boxes in [x, y, w, h] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(0, box1[2]) * max(0, box1[3])
    box2_area = max(0, box2[2]) * max(0, box2[3])
    union_area = box1_area + box2_area - inter_area
    return inter_area / (union_area + 1e-6)


def linear_assignment(cost_matrix, threshold=0.5):
    """Optimal Hungarian assignment filtered by maximum allowed cost."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    matched_rows = set()
    matched_cols = set()
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] <= threshold:
            matches.append((int(i), int(j)))
            matched_rows.add(int(i))
            matched_cols.add(int(j))

    unmatched_rows = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
    unmatched_cols = [j for j in range(cost_matrix.shape[1]) if j not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


class TrackEval:
    """
    Lightweight MOT-style evaluator for coursework/demo use.

    It computes IDF1/IDP/IDR from detection matches and a simplified MOTA:
        MOTA = 1 - (FN + FP + IDSW) / GT_detections
    """

    def __init__(self, if_kuajing=False):
        self.if_kuajing = if_kuajing
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.id_switches = 0
        self.gt_detections = 0
        self.pr_detections = 0
        self.frame_matches = {}
        self.track_assignments = {}

    def _process_detection_csv(self, df, id_field='identity_id', mark_id=0):
        mem = {}
        if id_field not in df.columns:
            fallback = 'identity_id' if 'identity_id' in df.columns else 'primary_uuid'
            if fallback not in df.columns:
                raise ValueError(f"ID field '{id_field}' not found and no fallback ID column exists")
            id_field = fallback

        for _, row in df.iterrows():
            idx_frame = int(row['idx_frame'])
            box_x1 = float(row['box_x1'])
            box_y1 = float(row['box_y1'])
            box_w = float(row['box_x2']) - float(row['box_x1'])
            box_h = float(row['box_y2']) - float(row['box_y1'])
            track_id = row[id_field]
            camera_id = row.get('camera_id', '1')
            camera_name = row.get('camera_name', 'scene1')

            if mark_id == 0:
                mark = f"{camera_id}_{camera_name}_{idx_frame}"
            elif mark_id == 1:
                mark = str(idx_frame)
            else:
                raise ValueError(f"Unsupported mark_id: {mark_id}")

            mem.setdefault(mark, []).append([box_x1, box_y1, box_w, box_h, track_id])
        return mem

    def _process_detection_txt(self, df, mark_id=0):
        mem = {}
        for _, row in df.iterrows():
            idx_frame = int(row['frame_number'])
            mark = str(idx_frame)
            mem.setdefault(mark, []).append([
                float(row['left']), float(row['top']), float(row['width']), float(row['height']), row['identity_id']
            ])
        return mem

    def _process_detection(self, df, suffix, id_field, mark):
        if suffix == 'txt':
            return self._process_detection_txt(df, mark)
        if suffix in {'csv', 'xlsx'}:
            return self._process_detection_csv(df, id_field, mark)
        raise ValueError(f"Unsupported suffix: {suffix}")

    def evaluate(self, pr_path, gt_path, id_field='identity_id', mark_id=1, iou_threshold=0.5):
        self.reset()
        pr_df = read_file(pr_path, id_field)
        gt_df = read_file(gt_path, id_field)
        pr_suffix = pr_path.split('.')[-1]
        gt_suffix = gt_path.split('.')[-1]

        pr_id = 'secondary_uuid' if self.if_kuajing and 'secondary_uuid' in pr_df.columns else id_field
        mem_pr = self._process_detection(pr_df, pr_suffix, pr_id, mark_id)
        mem_gt = self._process_detection(gt_df, gt_suffix, id_field, mark_id)

        self.gt_detections = sum(len(v) for v in mem_gt.values())
        self.pr_detections = sum(len(v) for v in mem_pr.values())

        matched_pr_keys = set()
        for frame in sorted(mem_gt.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            gt_items = mem_gt[frame]
            pr_items = mem_pr.get(frame, [])
            if frame in mem_pr:
                matched_pr_keys.add(frame)

            if not pr_items:
                self.fn += len(gt_items)
                continue

            gt_boxes = np.array(gt_items)[:, :4].astype(float)
            pr_boxes = np.array(pr_items)[:, :4].astype(float)
            gt_ids = [item[-1] for item in gt_items]
            pr_ids = [item[-1] for item in pr_items]

            cost_matrix = np.ones((len(gt_boxes), len(pr_boxes))) * 2.0
            for i in range(len(gt_boxes)):
                for j in range(len(pr_boxes)):
                    cost_matrix[i, j] = 1 - iou(gt_boxes[i], pr_boxes[j])

            matches, unmatched_gt, unmatched_pr = linear_assignment(cost_matrix, threshold=1 - iou_threshold)
            self.tp += len(matches)
            self.fn += len(unmatched_gt)
            self.fp += len(unmatched_pr)

            self.frame_matches[frame] = []
            for i, j in matches:
                gt_id = gt_ids[i]
                pr_id_value = pr_ids[j]
                self.frame_matches[frame].append((gt_id, pr_id_value))
                if gt_id in self.track_assignments and self.track_assignments[gt_id] != pr_id_value:
                    self.id_switches += 1
                self.track_assignments[gt_id] = pr_id_value

        # predictions on frames with no GT are false positives
        for pk, items in mem_pr.items():
            if pk not in matched_pr_keys:
                self.fp += len(items)

    def get_result(self):
        IDP = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        IDR = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        IDF1 = 2 * IDP * IDR / (IDP + IDR) if (IDP + IDR) > 0 else 0
        MOTA = 1 - (self.fn + self.fp + self.id_switches) / self.gt_detections if self.gt_detections > 0 else 0

        result = {
            'IDF1': round(IDF1 * 100, 2),
            'IDP': round(IDP * 100, 2),
            'IDR': round(IDR * 100, 2),
            'MOTA': round(MOTA * 100, 2),
            'IDs': int(self.id_switches),
            'GTs': int(self.gt_detections),
            'TP': int(self.tp),
            'FP': int(self.fp),
            'FN': int(self.fn),
        }

        table = PrettyTable()
        table.field_names = ['IDF1(up)', 'IDP(up)', 'IDR(up)', 'MOTA(up)', 'IDs(down)', 'GT dets']
        table.add_row([result['IDF1'], result['IDP'], result['IDR'], result['MOTA'], result['IDs'], result['GTs']])
        print(table)
        print('\nDetailed Results:')
        print(f"True Positives: {result['TP']}")
        print(f"False Positives: {result['FP']}")
        print(f"False Negatives: {result['FN']}")
        return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True, help='Prediction results CSV file')
    parser.add_argument('--gt_file', type=str, required=True, help='Ground truth file')
    parser.add_argument('--id_field', type=str, default='identity_id', help='ID field name')
    args = parser.parse_args()

    evaluator = TrackEval()
    evaluator.evaluate(args.pred_file, args.gt_file, id_field=args.id_field, mark_id=1)
    evaluator.get_result()

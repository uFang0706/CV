import ast
import os
import warnings

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', message='The behavior of DataFrame concatenation with empty or all-NA entries is deprecated')


def get_rotated_rect(keypoints, keypoints_conf):
    upper_pairs = [(5, 6), (1, 2), (3, 4), (0, 0)]
    lower_pairs = [(11, 12), (15, 16), (13, 14)]
    keypoints_conf_threshold = 0.4

    valid_keypoints = []
    for kp, conf in zip(keypoints, keypoints_conf):
        if conf > keypoints_conf_threshold and not (kp[0] == 0 and kp[1] == 0):
            valid_keypoints.append(kp)

    if len(valid_keypoints) < 2:
        return None

    upper_center = None
    for kp1, kp2 in upper_pairs:
        if (keypoints_conf[kp1] > keypoints_conf_threshold and keypoints_conf[kp2] > keypoints_conf_threshold and
                not (keypoints[kp1][0] == 0 and keypoints[kp1][1] == 0) and
                not (keypoints[kp2][0] == 0 and keypoints[kp2][1] == 0)):
            upper_center = (keypoints[kp1] + keypoints[kp2]) / 2
            break

    lower_center = None
    for kp1, kp2 in lower_pairs:
        if (keypoints_conf[kp1] > keypoints_conf_threshold and keypoints_conf[kp2] > keypoints_conf_threshold and
                not (keypoints[kp1][0] == 0 and keypoints[kp1][1] == 0) and
                not (keypoints[kp2][0] == 0 and keypoints[kp2][1] == 0)):
            lower_center = (keypoints[kp1] + keypoints[kp2]) / 2
            break

    if upper_center is None or lower_center is None:
        return None

    direction = lower_center - upper_center
    angle = float(np.arctan2(direction[1], direction[0]) * 180 / np.pi)
    points = np.array([valid_keypoints], dtype=np.float32)
    rect = cv2.minAreaRect(points[0])

    if rect[1][0] > rect[1][1]:
        return None
    return rect[0], rect[1], angle - 90


def calculate_rotated_rect_vertices(center, size, angle):
    angle_rad = np.radians(angle)
    width, height = size
    x, y = center
    half_width = width / 2
    half_height = height / 2
    offsets = [(-half_width, -half_height), (half_width, -half_height),
               (half_width, half_height), (-half_width, half_height)]
    vertices = []
    for offset_x, offset_y in offsets:
        rotated_x = offset_x * np.cos(angle_rad) - offset_y * np.sin(angle_rad)
        rotated_y = offset_x * np.sin(angle_rad) + offset_y * np.cos(angle_rad)
        vertices.append((x + rotated_x, y + rotated_y))
    return vertices


class Visualizer:
    def __init__(self):
        self.record_mem = {}
        self.global_count = 0

    @staticmethod
    def _parse_pose_points(value):
        """Safely parse optional pose keypoints from CSV; empty/NaN values are ignored."""
        if pd.isna(value):
            return None
        value = str(value).strip()
        if not value:
            return None
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        if not isinstance(parsed, (list, tuple)) or len(parsed) < 8:
            return None
        return parsed[:8]

    def _process_detection(self, df, id_field='identity_id'):
        mem = {}
        if id_field not in df.columns:
            id_field = 'identity_id' if 'identity_id' in df.columns else 'primary_uuid'

        for _, row in df.iterrows():
            idx_frame = int(row['idx_frame'])
            box_x1 = float(row['box_x1'])
            box_y1 = float(row['box_y1'])
            box_w = float(row['box_x2']) - float(row['box_x1'])
            box_h = float(row['box_y2']) - float(row['box_y1'])
            primary_uuid = row[id_field]

            if primary_uuid not in self.record_mem:
                self.record_mem[primary_uuid] = self.global_count
                self.global_count += 1

            mark = row['frame'] if 'frame' in row and pd.notna(row['frame']) else f"scene1_{idx_frame:03d}.jpg"
            if mark not in mem:
                mem[mark] = []

            pose = self._parse_pose_points(row['xys']) if 'xys' in row else None
            item = [box_x1, box_y1, box_w, box_h, self.record_mem[primary_uuid]]
            if pose is not None:
                item.extend(pose)
            mem[mark].append(item)
        return mem

    @staticmethod
    def _blank_canvas(items, width=640, height=480):
        """Create a synthetic canvas when demo images are not shipped in the repo."""
        if items:
            arr = np.array(items)[:, :4].astype(float)
            width = max(width, int(np.max(arr[:, 0] + arr[:, 2]) + 60))
            height = max(height, int(np.max(arr[:, 1] + arr[:, 3]) + 60))
        image = np.full((height, width, 3), 245, dtype=np.uint8)
        cv2.putText(image, 'synthetic demo frame', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 90, 90), 2)
        return image

    def visualize(self, csv_path, output_dir=None, id_field='identity_id'):
        csv_df = pd.read_csv(csv_path)
        mem_csv = self._process_detection(csv_df, id_field)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        saved = 0
        for frame_name, items in mem_csv.items():
            image = cv2.imread(frame_name) if os.path.exists(str(frame_name)) else None
            if image is None:
                image = self._blank_canvas(items)

            for item in items:
                x1, y1, w, h, track_id = item[:5]
                x1, y1, w, h = int(x1), int(y1), int(w), int(h)
                cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (37, 138, 255), 2)
                cv2.putText(image, f"ID:{track_id}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (37, 138, 255), 2)
                if len(item) > 5:
                    pose_xy = np.array(item[5:]).reshape(-1, 2)
                    for x, y in pose_xy:
                        if x != 0 and y != 0:
                            cv2.circle(image, (int(x), int(y)), 4, (50, 180, 50), -1)

            if output_dir is not None:
                safe_name = os.path.basename(str(frame_name)).replace(os.sep, '_')
                output_path = os.path.join(output_dir, f"vis_{safe_name}")
                cv2.imwrite(output_path, image)
                saved += 1
            else:
                cv2.imshow('Visualization', image)
                cv2.waitKey(0)

        if output_dir is None:
            cv2.destroyAllWindows()
        print(f"Visualization completed. Saved {saved} frames to {output_dir}" if output_dir else "Visualization completed.")

    def visualize_tracking_result(self, image, boxes, track_ids, pose_points=None, output_path=None):
        for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
            x1, y1, w, h = map(int, box)
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(image, (x1, y1), (x2, y2), (37, 138, 255), 2)
            cv2.putText(image, f"ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (37, 138, 255), 2)
            if pose_points is not None and i < len(pose_points):
                for x, y in pose_points[i]:
                    if x > 0 and y > 0:
                        cv2.circle(image, (int(x), int(y)), 3, (50, 180, 50), -1)
        if output_path:
            cv2.imwrite(output_path, image)
        else:
            cv2.imshow('Tracking Result', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, required=True, help='CSV file with tracking results')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for visualizations')
    parser.add_argument('--id_field', type=str, default='identity_id', help='ID field name')
    args = parser.parse_args()

    model = Visualizer()
    model.visualize(args.csv_file, args.output_dir, args.id_field)

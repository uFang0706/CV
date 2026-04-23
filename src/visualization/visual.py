import pandas as pd
import numpy as np
import warnings
import cv2
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
    for pair in upper_pairs:
        kp1, kp2 = pair
        if (keypoints_conf[kp1] > keypoints_conf_threshold and keypoints_conf[kp2] > keypoints_conf_threshold and
            not (keypoints[kp1][0] == 0 and keypoints[kp1][1] == 0) and
            not (keypoints[kp2][0] == 0 and keypoints[kp2][1] == 0)):
            upper_center = (keypoints[kp1] + keypoints[kp2]) / 2
            break

    lower_center = None
    for pair in lower_pairs:
        kp1, kp2 = pair
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

    offsets = [
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height)
    ]

    vertices = []
    for offset_x, offset_y in offsets:
        rotated_x = offset_x * np.cos(angle_rad) - offset_y * np.sin(angle_rad)
        rotated_y = offset_x * np.sin(angle_rad) + offset_y * np.cos(angle_rad)
        vertex_x = x + rotated_x
        vertex_y = y + rotated_y
        vertices.append((vertex_x, vertex_y))

    return vertices


class Visualizer:
    def __init__(self):
        self.record_mem = {}
        self.global_count = 0

    def _process_detection(self, df, id_field='primary_uuid'):
        mem = {}
        for index, row in df.iterrows():
            idx_frame = row['idx_frame']
            box_x1 = row['box_x1']
            box_y1 = row['box_y1']
            box_w = row['box_x2'] - row['box_x1']
            box_h = row['box_y2'] - row['box_y1']
            primary_uuid = row[id_field]
            camera_id = row['camera_id']
            camera_name = row['camera_name']

            if primary_uuid not in self.record_mem:
                self.record_mem[primary_uuid] = self.global_count
                self.global_count += 1

            mark = row['frame']
            if mark not in mem:
                mem[mark] = []

            if 'xys' in row:
                xys = eval(row['xys'])
                pose_x1, pose_y1 = xys[0], xys[1]
                pose_x2, pose_y2 = xys[2], xys[3]
                pose_x3, pose_y3 = xys[4], xys[5]
                pose_x4, pose_y4 = xys[6], xys[7]
                mem[mark].append([box_x1, box_y1, box_w, box_h, self.record_mem[primary_uuid],
                                 pose_x1, pose_y1, pose_x2, pose_y2, pose_x3, pose_y3, pose_x4, pose_y4])
            else:
                mem[mark].append([box_x1, box_y1, box_w, box_h, self.record_mem[primary_uuid]])
        return mem

    def visualize(self, csv_path, output_dir=None, id_field='primary_uuid'):
        csv_df = read_csv(csv_path)
        mem_csv = self._process_detection(csv_df, id_field)

        if output_dir is not None:
            import os
            os.makedirs(output_dir, exist_ok=True)

        for k in mem_csv:
            filename = k
            prx1y1wh = np.array(mem_csv[k])[:, :4]
            pr_id = np.array(mem_csv[k])[:, 4].tolist()
            pose_xys = np.array(mem_csv[k])[:, 5:]

            image = cv2.imread(filename)
            if image is None:
                print(f"无法读取图像: {filename}")
                continue

            for i, (x1, y1, w, h) in enumerate(prx1y1wh):
                x1, y1, w, h = int(x1), int(y1), int(w), int(h)
                cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.putText(image, str(pr_id[i]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            for pose_xy in pose_xys:
                pose_xy = pose_xy.reshape(-1, 2)
                for x, y in pose_xy:
                    if x != 0 and y != 0:
                        x, y = int(x), int(y)
                        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

            if output_dir is not None:
                output_path = os.path.join(output_dir, f"vis_{k}.jpg")
                cv2.imwrite(output_path, image)
            else:
                cv2.imshow('Visualization', image)
                cv2.waitKey(0)

        cv2.destroyAllWindows()

    def visualize_tracking_result(self, image, boxes, track_ids, pose_points=None, output_path=None):
        for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
            x1, y1, w, h = map(int, box)
            x2, y2 = x1 + w, y1 + h

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"ID:{track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if pose_points is not None and i < len(pose_points):
                for x, y in pose_points[i]:
                    if x > 0 and y > 0:
                        cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)

        if output_path:
            cv2.imwrite(output_path, image)
        else:
            cv2.imshow('Tracking Result', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True, help="CSV file with tracking results")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for visualizations")
    parser.add_argument("--id_field", type=str, default='primary_uuid', help="ID field name")
    args = parser.parse_args()

    model = Visualizer()
    model.visualize(args.csv_file, args.output_dir, args.id_field)

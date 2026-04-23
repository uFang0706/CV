import cv2
import numpy as np
import copy


def box_crop(im, dets):
    crops = []
    for tlbr in dets:
        x1, y1, w, h = map(int, tlbr)
        x2, y2 = x1 + w // 2, y1 + h // 2
        x1, y1 = x1 - w // 2, y1 - h // 2
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(im.shape[1], x2), min(im.shape[0], y2)
        crop = im[y1:y2, x1:x2]
        crops.append(crop)
    return crops


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
        vertices.append([vertex_x, vertex_y])

    return vertices


def get_rotated_rect(keypoints, keypoints_conf):
    upper_pairs = [(5, 6), (1, 2), (3, 4), (0, 0)]
    lower_pairs = [(11, 12), (15, 16), (13, 14)]
    keypoints_conf_threshold = 0.1

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


def crop_rotated_rect(image, rect):
    center, size, angle = rect
    angle = float(angle)
    new_rect = (center, (size[0] * 1., size[1] * 1.), angle)
    center, size, angle = new_rect

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    x = int(center[0] - size[0] / 2)
    y = int(center[1] - size[1] / 2)
    w = int(size[0])
    h = int(size[1])

    y = int(y - 0.1 * h)

    x = max(0, min(x, image.shape[1] - 1))
    y = max(0, min(y, image.shape[0] - 1))
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    cropped = rotated[y:y + h, x:x + w]

    return cropped


def get_rotated_rect_pca_manual(keypoints, keypoints_conf, conf_threshold=0.2, exclude_indices=[5, 6, 7, 8, 9, 10]):
    valid = [(pt, conf) for idx, (pt, conf) in enumerate(zip(keypoints, keypoints_conf))
             if conf > conf_threshold and not (pt[0] == 0 and pt[1] == 0)]

    if len(valid) < 3:
        return None

    pts = np.array([pt for pt, _ in valid], dtype=np.float32)
    mean = np.mean(pts, axis=0)
    centered = pts - mean

    _, _, vt = np.linalg.svd(centered)
    direction = vt[0]

    angle = np.arctan2(direction[1], direction[0]) * 180. / np.pi
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    cos = np.cos(np.radians(-angle))
    sin = np.sin(np.radians(-angle))

    rotated_pts = []
    for pt in pts:
        dx = pt[0] - mean[0]
        dy = pt[1] - mean[1]
        x_rot = dx * cos - dy * sin
        y_rot = dx * sin + dy * cos
        rotated_pts.append([x_rot + mean[0], y_rot + mean[1]])

    rotated_pts = np.array(rotated_pts, dtype=np.float32)

    x_min, y_min = np.min(rotated_pts, axis=0)
    x_max, y_max = np.max(rotated_pts, axis=0)

    rect_center_rotated = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    width = x_max - x_min
    height = y_max - y_min

    half_w, half_h = width / 2, height / 2
    corners_rotated = np.array([
        [rect_center_rotated[0] - half_w, rect_center_rotated[1] - half_h],
        [rect_center_rotated[0] + half_w, rect_center_rotated[1] - half_h],
        [rect_center_rotated[0] + half_w, rect_center_rotated[1] + half_h],
        [rect_center_rotated[0] - half_w, rect_center_rotated[1] + half_h],
    ])

    cos_reverse = np.cos(np.radians(angle))
    sin_reverse = np.sin(np.radians(angle))

    corners_original = []
    for corner in corners_rotated:
        dx = corner[0] - mean[0]
        dy = corner[1] - mean[1]
        x_orig = dx * cos_reverse - dy * sin_reverse
        y_orig = dx * sin_reverse + dy * cos_reverse
        corners_original.append([x_orig + mean[0], y_orig + mean[1]])

    corners_original = np.array(corners_original, dtype=np.float32)

    x_min_orig, y_min_orig = np.min(corners_original, axis=0)
    x_max_orig, y_max_orig = np.max(corners_original, axis=0)

    final_width = x_max_orig - x_min_orig
    final_height = y_max_orig - y_min_orig
    final_center = np.array([(x_min_orig + x_max_orig) / 2, (y_min_orig + y_max_orig) / 2])

    return (float(final_center[0]), float(final_center[1])), (float(final_width), float(final_height)), angle


def draw_rotated_rect2(image, rect):
    center, size, angle = rect
    if size[0] <= 0 or size[1] <= 0:
        return image, None

    size_int = (int(1.15 * size[0]), int(1.25 * size[1]))

    box = cv2.boxPoints((center, size_int, angle))
    M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
    box = cv2.transform(np.array([box]), M_inv)[0]

    width, height = size_int
    src_pts = np.array([
        box[2],
        box[3],
        box[0],
        box[1]
    ], dtype="float32")
    dst_pts = np.array([
        [0, height - 1],
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped = cv2.warpPerspective(image, M, (width, height))

    return cropped


def pose_crop(im, dets, model):
    crops = []

    xc = dets[:, 0]
    yc = dets[:, 1]
    w = dets[:, 2]
    h = dets[:, 3]

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    convert_dets = np.column_stack((x1, y1, x2, y2))
    key_points, key_scores = model.multi_inference(im, convert_dets, 0.1)

    new_im = copy.deepcopy(im)

    for index, (key_point, key_score) in enumerate(zip(key_points, key_scores)):
        rect = get_rotated_rect_pca_manual(key_point, key_score)
        x1, y1, w, h = map(int, dets[index])
        x2, y2 = x1 + w // 2, y1 + h // 2
        x1, y1 = x1 - w // 2, y1 - h // 2

        if w * h < 100 * 200 or rect is None:
            crop = im[y1:y2, x1:x2]
            crops.append(crop)
            continue
        try:
            crop = draw_rotated_rect2(im, rect)[0]
            c_w = crop.shape[1]
            c_h = crop.shape[0]
        except:
            import pdb
            pdb.set_trace()

        if c_h / c_w > 5:
            crop = im[y1:y2, x1:x2]
            crops.append(crop)
            continue

        crops.append(crop)

    return crops


class CropStrategy:
    def __init__(self, strategy='box', pose_model=None):
        self.strategy = strategy
        self.pose_model = pose_model

    def crop(self, image, detections):
        if self.strategy == 'box':
            return box_crop(image, detections)
        elif self.strategy == 'pose':
            if self.pose_model is None:
                raise ValueError("Pose model is required for pose-based cropping")
            return pose_crop(image, detections, self.pose_model)
        else:
            raise ValueError(f"Unknown cropping strategy: {self.strategy}")

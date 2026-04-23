import os
import numpy as np
import pandas as pd
import yaml
from pathlib import Path


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path):
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)


def read_csv(csv_path):
    return pd.read_csv(csv_path)


def write_csv(df, csv_path):
    df.to_csv(csv_path, index=False)


def read_results(path, data_type, is_gt=False, is_ignore=False):
    if data_type == 'mot':
        return read_mot_results(path, is_gt, is_ignore)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def read_mot_results(path, is_gt, is_ignore):
    if not os.path.exists(path):
        return {}

    results = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue

            frame_id = int(parts[0])
            track_id = int(parts[1])
            bb_left = float(parts[2])
            bb_top = float(parts[3])
            bb_width = float(parts[4])
            bb_height = float(parts[5])
            conf = float(parts[6]) if not is_gt else 1.0
            cls = int(parts[7]) if not is_gt and len(parts) > 7 else 1

            if is_ignore and cls != 2:
                continue

            if frame_id not in results:
                results[frame_id] = []

            results[frame_id].append({
                'track_id': track_id,
                'bb_left': bb_left,
                'bb_top': bb_top,
                'bb_width': bb_width,
                'bb_height': bb_height,
                'conf': conf,
                'cls': cls
            })

    return results


def unzip_objs(objs):
    if len(objs) == 0:
        return [], [], [], []

    tlwhs = []
    ids = []
    confs = []
    clss = []

    for obj in objs:
        tlwhs.append([obj['bb_left'], obj['bb_top'], obj['bb_width'], obj['bb_height']])
        ids.append(obj['track_id'])
        confs.append(obj['conf'])
        clss.append(obj['cls'])

    return tlwhs, ids, confs, clss


def tlwh_to_tlbr(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[2:] += ret[:2]
    return ret


def tlbr_to_tlwh(tlbr):
    ret = np.asarray(tlbr).copy()
    ret[2:] -= ret[:2]
    return ret


def xcycwh_to_tlbr(xcycwh):
    ret = np.asarray(xcycwh).copy()
    ret[0] = xcycwh[0] - xcycwh[2] / 2
    ret[1] = xcycwh[1] - xcycwh[3] / 2
    ret[2] = xcycwh[0] + xcycwh[2] / 2
    ret[3] = xcycwh[1] + xcycwh[3] / 2
    return ret


def tlwh_to_xcycwh(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    return ret

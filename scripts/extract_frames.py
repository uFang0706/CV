#!/usr/bin/env python
"""
从Wuzhou监控视频中提取关键帧，用于生成带真实背景的配图
"""

import cv2
import os
import numpy as np

def extract_key_frames(video_path, output_dir, intervals=[100, 300, 500, 800, 1000]):
    """从视频中提取指定帧"""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")

    extracted = []
    for frame_idx in intervals:
        if frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
                cv2.imwrite(output_path, frame)
                extracted.append((frame_idx, output_path))
                print(f"Extracted frame {frame_idx} -> {output_path}")

    cap.release()
    return extracted


def extract_frame_with_tracking(video_path, gt_file, output_dir, frame_idx=300):
    """提取带跟踪结果的特定帧"""
    os.makedirs(output_dir, exist_ok=True)

    import pandas as pd
    gt_df = pd.read_csv(gt_file, names=['frame', 'id', 'x', 'y', 'w', 'h', 'c', 'c2', 'c3'])

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if not ret:
        print(f"Failed to read frame {frame_idx}")
        return None

    frame_gt = gt_df[gt_df['frame'] == frame_idx + 1]

    for _, row in frame_gt.iterrows():
        x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        track_id = int(row['id'])

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        color = colors[track_id % len(colors)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"ID:{track_id}", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_path = os.path.join(output_dir, f"tracking_frame_{frame_idx:04d}.png")
    cv2.imwrite(output_path, frame)
    print(f"Saved tracking visualization: {output_path}")

    cap.release()
    return output_path


if __name__ == '__main__':
    video_path = 'original_project/test_videos/Wuzhou_MidRoad/Wuzhou_MidRoad.mp4'
    gt_file = 'original_project/test_videos/Wuzhou_MidRoad/gt.txt'
    output_dir = 'report/figures/frames'

    print("=" * 50)
    print("Extracting key frames from surveillance video")
    print("=" * 50)

    intervals = [50, 150, 300, 500, 750, 1000, 1250, 1500, 1750, 2000]
    extract_key_frames(video_path, output_dir, intervals)

    extract_frame_with_tracking(video_path, gt_file, output_dir, frame_idx=300)

    print("=" * 50)
    print("Frame extraction complete!")
    print(f"Output directory: {output_dir}")

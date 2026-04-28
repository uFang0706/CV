#!/usr/bin/env python
"""
整合原始项目数据运行脚本
整合了原始monoscenetrackeval项目的:
- MOT17/MOT20 Ground Truth数据
- Wuzhou_MidRoad跟踪结果和GT
用于评估和验证跟踪性能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from datetime import datetime
import pandas as pd
from src.evaluation.metric import TrackEval


def read_mot_txt(path):
    """读取MOT格式txt文件"""
    return pd.read_csv(
        path,
        sep=',',
        names=['frame_number', 'identity_id', 'left', 'top', 'width', 'height', 'score', 'x', 'y', 'z']
    )


def evaluate_wuzhou():
    """评估Wuzhou_MidRoad跟踪结果"""
    print("=" * 60)
    print("评估 Wuzhou_MidRoad 跟踪结果")
    print("=" * 60)

    gt_file = 'original_project/test_videos/Wuzhou_MidRoad/gt.txt'
    pred_file = 'original_project/results/trackers/Wuzhou_MidRoad.txt'

    if not os.path.exists(gt_file):
        print(f"错误: GT文件不存在: {gt_file}")
        return None

    if not os.path.exists(pred_file):
        print(f"错误: 预测文件不存在: {pred_file}")
        return None

    print(f"GT文件: {gt_file}")
    print(f"预测文件: {pred_file}")

    gt_df = read_mot_txt(gt_file)
    pred_df = read_mot_txt(pred_file)

    print(f"\nGT数据: {len(gt_df)} 条记录, {gt_df['frame_number'].max()} 帧, {gt_df['identity_id'].nunique()} 个ID")
    print(f"预测数据: {len(pred_df)} 条记录, {pred_df['frame_number'].max()} 帧, {pred_df['identity_id'].nunique()} 个ID")

    evaluator = TrackEval()
    evaluator.evaluate(pred_file, gt_file, id_field='identity_id', mark_id=1)
    results = evaluator.get_result()

    return results


def evaluate_mot17_sample():
    """评估MOT17示例数据"""
    print("\n" + "=" * 60)
    print("评估 MOT17 示例数据")
    print("=" * 60)

    gt_dir = 'original_project/results/gt/MOT17-val'
    if not os.path.exists(gt_dir):
        print(f"注意: MOT17 GT目录不存在: {gt_dir}")
        return None

    seqs = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-09-FRCNN']
    total_gt = 0
    total_pred = 0

    for seq in seqs:
        gt_file = os.path.join(gt_dir, seq, 'gt', 'gt.txt')
        if os.path.exists(gt_file):
            gt_df = read_mot_txt(gt_file)
            total_gt += len(gt_df)
            print(f"  {seq}: {len(gt_df)} GT记录")

    print(f"\nMOT17示例 GT总数: {total_gt} 条记录")
    return {'message': 'MOT17 GT数据已就绪'}


def main():
    parser = argparse.ArgumentParser(description='整合原始项目运行脚本')
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', 'wuzhou', 'mot17'],
                        help='选择运行的实验')
    parser.add_argument('--output', type=str,
                        default='original_project/run_logs',
                        help='输出日志目录')
    parser.add_argument('--save_results', action='store_true',
                        help='保存评估结果')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.output, f'run_log_{timestamp}.txt')

    print(f"\n运行时间: {datetime.now()}")
    print(f"日志文件: {log_file}")

    results = {}

    if args.exp in ['all', 'wuzhou']:
        wuzhou_results = evaluate_wuzhou()
        if wuzhou_results:
            results['wuzhou'] = wuzhou_results

    if args.exp in ['all', 'mot17']:
        mot17_results = evaluate_mot17_sample()
        if mot17_results:
            results['mot17'] = mot17_results

    print("\n" + "=" * 60)
    print("运行完成")
    print("=" * 60)

    if results.get('wuzhou'):
        r = results['wuzhou']
        print(f"\nWuzhou_MidRoad 评估结果:")
        print(f"  IDF1: {r['IDF1']:.2f}%")
        print(f"  IDP: {r['IDP']:.2f}%")
        print(f"  IDR: {r['IDR']:.2f}%")
        print(f"  ID Switches: {r['IDs']}")
        print(f"  GT Objects: {r['GTs']}")

    if args.save_results:
        result_file = os.path.join(args.output, f'results_{timestamp}.txt')
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"Run Time: {datetime.now()}\n")
            f.write("=" * 60 + "\n")
            for exp_name, exp_results in results.items():
                f.write(f"\n{exp_name}:\n")
                if isinstance(exp_results, dict):
                    for k, v in exp_results.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"  {exp_results}\n")
        print(f"\nResults saved to: {result_file}")

    return results


if __name__ == '__main__':
    main()

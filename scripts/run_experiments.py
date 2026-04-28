#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pandas as pd
import numpy as np
from src.evaluation.metric import TrackEval
from src.visualization.visual import Visualizer


EXPERIMENTS = {
    'baseline': {
        'name': 'Baseline (YOLO + DeepSORT)',
        'description': 'Basic detection + tracking without ReID features'
    },
    'box_crop': {
        'name': 'Box Crop ReID',
        'description': 'Standard bounding box cropping for ReID'
    },
    'pose_crop': {
        'name': 'Pose-based Cropping',
        'description': 'Human pose estimation-based cropping'
    },
    'kalman': {
        'name': 'Kalman Filter Acceleration',
        'description': 'Kalman filter for motion prediction'
    }
}


def run_single_experiment(exp_name, pred_file, gt_file=None):
    print(f"\n{'=' * 60}")
    print(f"Running Experiment: {exp_name}")
    print(f"{'=' * 60}")

    if exp_name not in EXPERIMENTS:
        print(f"Error: Unknown experiment {exp_name}")
        return None

    exp_info = EXPERIMENTS[exp_name]
    print(f"Description: {exp_info['description']}")

    if not os.path.exists(pred_file):
        print(f"Error: Prediction file not found: {pred_file}")
        return None

    evaluator = TrackEval()
    if gt_file and os.path.exists(gt_file):
        evaluator.evaluate(pred_file, gt_file, id_field='identity_id', mark_id=1)
        results = evaluator.get_result()
    else:
        print("Note: No ground truth file, skipping full evaluation")
        results = {'IDF1': 0, 'MOTA': 0, 'IDs': 0, 'GTs': 0}

    return results


def run_all_experiments():
    print("Running All Experiments")
    print("=" * 60)

    results = {}
    for exp_name in EXPERIMENTS.keys():
        pred_file = f'data/demo_results/{exp_name}_results.csv'
        gt_file = 'data/sample_csv/sample_gt.csv'

        results[exp_name] = run_single_experiment(exp_name, pred_file, gt_file)

    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)

    print(f"{'Experiment':<20} {'IDF1':>10} {'MOTA':>10} {'IDs':>10}")
    print("-" * 60)
    for exp_name, res in results.items():
        if res:
            print(f"{exp_name:<20} {res['IDF1']:>10.2f} {res['MOTA']:>10.2f} {res['IDs']:>10}")
        else:
            print(f"{exp_name:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run tracking experiments')
    parser.add_argument('--exp', type=str, default='all',
                        help='Experiment name or "all"')
    parser.add_argument('--pred_file', type=str,
                        default='data/demo_results/pose_crop_results.csv',
                        help='Prediction file')
    parser.add_argument('--gt_file', type=str,
                        default='data/sample_csv/sample_gt.csv',
                        help='Ground truth file')
    parser.add_argument('--output', type=str,
                        default='results/metrics/experiment_results.txt',
                        help='Output file')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.exp == 'all':
        results = run_all_experiments()
    else:
        results = {args.exp: run_single_experiment(args.exp, args.pred_file, args.gt_file)}

    with open(args.output, 'w') as f:
        f.write("Experiment Results Summary\n")
        f.write("=" * 60 + "\n")
        for exp_name, res in results.items():
            if res:
                f.write(f"\n{exp_name}:\n")
                f.write(f"  IDF1: {res['IDF1']}\n")
                f.write(f"  MOTA: {res['MOTA']}\n")
                f.write(f"  IDs: {res['IDs']}\n")

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

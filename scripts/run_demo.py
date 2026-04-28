#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.metric import TrackEval
from src.visualization.visual import Visualizer
import argparse


def run_demo_mode():
    print("=" * 60)
    print("Running in DEMO mode - Metric Evaluation and Visualization")
    print("=" * 60)

    pred_file = 'data/demo_results/pose_crop_results.csv'
    gt_file = 'data/sample_csv/sample_gt.csv'

    print("\n1. Running Evaluation...")
    print("-" * 40)

    if not os.path.exists(pred_file):
        print(f"Error: Prediction file not found: {pred_file}")
        return

    if not os.path.exists(gt_file):
        print(f"Note: Ground truth file not found: {gt_file}")
        print("Creating synthetic ground truth for demo...")

        import pandas as pd
        df = pd.read_csv(pred_file)
        df['identity_id'] = df['identity_id']
        df.to_csv(gt_file, index=False)

    evaluator = TrackEval()
    evaluator.evaluate(pred_file, gt_file, id_field='primary_uuid', mark_id=1)
    results = evaluator.get_result()

    print("\n2. Running Visualization...")
    print("-" * 40)

    output_dir = 'results/visualizations'
    os.makedirs(output_dir, exist_ok=True)

    visualizer = Visualizer()
    print(f"Note: Visualization requires actual image files.")
    print(f"Results would be saved to: {output_dir}")

    print("\n3. Summary")
    print("-" * 40)
    print(f"IDF1: {results['IDF1']}")
    print(f"IDP: {results['IDP']}")
    print(f"IDR: {results['IDR']}")
    print(f"ID Switches: {results['IDs']}")
    print(f"Ground Truth Objects: {results['GTs']}")

    print("\n" + "=" * 60)
    print("Demo mode completed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='ReID-Enhanced Multi-Object Tracking Coursework')
    parser.add_argument('--mode', type=str, default='demo', choices=['full', 'demo'],
                        help='Run mode: full (with detection/tracking) or demo (metrics only)')
    parser.add_argument('--config', type=str, default='src/config/default.yaml',
                        help='Configuration file path')

    args = parser.parse_args()

    if args.mode == 'demo':
        run_demo_mode()
    elif args.mode == 'full':
        print("Full mode requires CUDA and model weights.")
        print("Please ensure you have:")
        print("  1. CUDA installed")
        print("  2. YOLO weights")
        print("  3. ReID model weights")
        print("Then run with appropriate configuration.")
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()

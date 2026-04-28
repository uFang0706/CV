#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.metric import TrackEval


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate tracking results')
    parser.add_argument('--pred_file', type=str,
                        default='data/sample_csv/sample_tracking_results.csv',
                        help='Prediction results CSV file')
    parser.add_argument('--gt_file', type=str,
                        default='data/sample_csv/sample_gt.csv',
                        help='Ground truth file')
    parser.add_argument('--id_field', type=str, default='identity_id',
                        help='ID field name')
    parser.add_argument('--output', type=str, default='results/metrics/eval_results.txt',
                        help='Output file for results')

    args = parser.parse_args()

    if not os.path.exists(args.pred_file):
        print(f"Error: Prediction file not found: {args.pred_file}")
        return

    if not os.path.exists(args.gt_file):
        print(f"Error: Ground truth file not found: {args.gt_file}")
        print("Note: Demo mode requires a ground truth file for evaluation.")
        return

    print(f"Evaluating prediction file: {args.pred_file}")
    print(f"Against ground truth file: {args.gt_file}")

    evaluator = TrackEval()
    evaluator.evaluate(args.pred_file, args.gt_file, id_field=args.id_field, mark_id=1)
    results = evaluator.get_result()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(f"IDF1: {results['IDF1']}\n")
        f.write(f"IDP: {results['IDP']}\n")
        f.write(f"IDR: {results['IDR']}\n")
        f.write(f"MOTA: {results['MOTA']}\n")
        f.write(f"ID Switches: {results['IDs']}\n")
        f.write(f"Ground Truth Detections: {results['GTs']}\n")

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

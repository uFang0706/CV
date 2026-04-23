#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.visualization.visual import Visualizer


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize tracking results')
    parser.add_argument('--csv_file', type=str,
                        default='data/sample_csv/sample_tracking_results.csv',
                        help='CSV file with tracking results')
    parser.add_argument('--output_dir', type=str,
                        default='results/visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--id_field', type=str, default='primary_uuid',
                        help='ID field name')

    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        return

    print(f"Visualizing tracking results from: {args.csv_file}")

    os.makedirs(args.output_dir, exist_ok=True)

    visualizer = Visualizer()
    visualizer.visualize(args.csv_file, args.output_dir, args.id_field)

    print(f"Visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

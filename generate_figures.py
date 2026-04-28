import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

os.makedirs('report/figures', exist_ok=True)

plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

IMG_FORMAT = 'png'
DPI = 150

def generate_system_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('ReID-assisted Multi-Object Tracking System Architecture', fontsize=14, fontweight='bold', pad=20)

    colors = ['#E3F2FD', '#E8F5E9', '#FFF3E0', '#F3E5F5', '#E0F7FA', '#FFFDE7']

    boxes = [
        {'pos': (0.5, 7), 'size': (2.5, 1.5), 'text': 'Input Video\nSequence', 'color': colors[0]},
        {'pos': (3.5, 7), 'size': (2.5, 1.5), 'text': 'YOLO\nDetector', 'color': colors[1]},
        {'pos': (6.5, 7), 'size': (2.5, 1.5), 'text': 'Kalman Filter\nPredictor', 'color': colors[2]},
        {'pos': (3.5, 4.5), 'size': (2.5, 1.5), 'text': 'Pose Estimation\n(HRNet)', 'color': colors[3]},
        {'pos': (6.5, 4.5), 'size': (2.5, 1.5), 'text': 'Cropping\nStrategy', 'color': colors[4]},
        {'pos': (9.5, 5.5), 'size': (2, 1.5), 'text': 'ReID Feature\nExtractor', 'color': colors[5]},
        {'pos': (3.5, 1.5), 'size': (2.5, 1.5), 'text': 'Data\nAssociation', 'color': colors[0]},
        {'pos': (6.5, 1.5), 'size': (2.5, 1.5), 'text': 'DeepSORT/\nBoostTrack', 'color': colors[1]},
        {'pos': (9.5, 1.5), 'size': (2, 1.5), 'text': 'Tracking\nResults', 'color': colors[2]},
    ]

    for box in boxes:
        rect = patches.FancyBboxPatch(box['pos'], box['size'][0], box['size'][1],
                                       boxstyle="round,pad=0.05,rounding_size=0.2",
                                       facecolor=box['color'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(box['pos'][0] + box['size'][0]/2, box['pos'][1] + box['size'][1]/2,
               box['text'], ha='center', va='center', fontsize=9, fontweight='bold')

    arrows = [
        (3, 7.75, 3.5, 7.75),
        (6, 7.75, 6.5, 7.75),
        (5.25, 4.5, 6.5, 4.5),
        (9, 5.5, 9.5, 5.5),
        (3, 5.25, 3.5, 5.25),
        (6, 3.75, 6.5, 2.25),
        (4.75, 4.5, 4.75, 2.25),
        (9, 4.5, 9.5, 2.25),
        (9, 2.25, 9.5, 2.25),
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    module_labels = [
        (2, 8.5, 'Detection Module', 'blue'),
        (7.5, 3, 'ReID Module', 'green'),
        (5, 0.5, 'Tracking Module', 'red'),
    ]

    for x, y, text, color in module_labels:
        ax.text(x, y, text, ha='center', va='center', fontsize=10, color=color, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'report/figures/system_architecture.{IMG_FORMAT}', format=IMG_FORMAT, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Generated: report/figures/system_architecture.{IMG_FORMAT}")


def generate_cropping_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for i, ax in enumerate(axes):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(['Original Detection Box', 'Standard Cropping', 'Pose-aware Cropping'][i],
                    fontsize=12, fontweight='bold')

        person_rect = patches.Rectangle((2, 1), 6, 8, linewidth=2,
                                        edgecolor='blue', facecolor='none', linestyle='--')
        ax.add_patch(person_rect)

        if i >= 1:
            crop_rect = patches.Rectangle((2.5, 1.5), 5, 7, linewidth=2,
                                          edgecolor='red', facecolor='lightyellow', alpha=0.5)
            ax.add_patch(crop_rect)

        if i == 2:
            keypoints = {
                'head': (5, 8),
                'neck': (5, 7),
                'left_shoulder': (3.5, 6),
                'right_shoulder': (6.5, 6),
                'left_hip': (4, 3.5),
                'right_hip': (6, 3.5),
            }
            for name, (kx, ky) in keypoints.items():
                circle = patches.Circle((kx, ky), 0.3, facecolor='orange', edgecolor='darkorange')
                ax.add_patch(circle)
                ax.text(kx, ky, name[0].upper(), ha='center', va='center', fontsize=6, fontweight='bold')

        ax.text(5, 0.3, ['Input: Bbox (x1,y1,x2,y2)', 'Output: Cropped Image', 'Output: Pose-optimized Crop'][i],
               ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(f'report/figures/cropping_comparison.{IMG_FORMAT}', format=IMG_FORMAT, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Generated: report/figures/cropping_comparison.{IMG_FORMAT}")


def generate_tracking_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    methods = ['Baseline', 'Box\nCrop', 'Kalman', 'Pose\nCrop']
    idf1_values = [97.44, 100.0, 100.0, 100.0]
    id_switches = [1, 1, 2, 0]

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    bars1 = ax1.bar(methods, idf1_values, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('IDF1 (%)', fontsize=11)
    ax1.set_title('IDF1 Comparison Across Methods', fontsize=12, fontweight='bold')
    ax1.set_ylim(94, 101)
    ax1.axhline(y=97.44, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    for bar, val in zip(bars1, idf1_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.legend()

    bars2 = ax2.bar(methods, id_switches, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('ID Switches', fontsize=11)
    ax2.set_title('ID Switches Comparison Across Methods', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 3)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    for bar, val in zip(bars2, id_switches):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f'{val}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'report/figures/tracking_comparison.{IMG_FORMAT}', format=IMG_FORMAT, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Generated: report/figures/tracking_comparison.{IMG_FORMAT}")


def generate_tracking_visualization():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Multi-Object Tracking Results Visualization', fontsize=14, fontweight='bold')

    tracks = [
        {'id': 1, 'color': 'red', 'points': [(2, 6), (4, 5), (6, 5.5), (8, 5), (10, 4.5), (12, 4), (14, 3.5)]},
        {'id': 2, 'color': 'blue', 'points': [(1, 3), (3, 4), (5, 5), (7, 6), (9, 7), (11, 8), (13, 9)]},
        {'id': 3, 'color': 'green', 'points': [(3, 10), (5, 9), (7, 8), (9, 7), (11, 6), (13, 5)]},
        {'id': 4, 'color': 'orange', 'points': [(2, 1), (4, 2), (6, 3), (8, 4), (10, 5), (12, 6)]},
        {'id': 5, 'color': 'purple', 'points': [(6, 11), (7, 9), (8, 8), (9, 7), (10, 6), (11, 5), (12, 4)]},
    ]

    for track in tracks:
        points = np.array(track['points'])
        ax.plot(points[:, 0], points[:, 1], color=track['color'], linewidth=2, alpha=0.6)

        for i, (x, y) in enumerate(track['points']):
            box = patches.FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8,
                                         boxstyle="round,pad=0.02,rounding_size=0.1",
                                         facecolor=track['color'], edgecolor='black', linewidth=1, alpha=0.8)
            ax.add_patch(box)
            ax.text(x, y, str(track['id']), ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white')

            if i < len(track['points']) - 1:
                next_point = track['points'][i+1]
                ax.annotate('', xy=(next_point[0]-0.4, next_point[1]),
                           xytext=(x+0.4, y),
                           arrowprops=dict(arrowstyle='->', color=track['color'],
                                         lw=1.5, alpha=0.5))

    legend_elements = [patches.Patch(facecolor=t['color'], edgecolor='black', label=f'ID {t["id"]}')
                      for t in tracks]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.text(8, 0.5, 'Frame Sequence: 1 → 2 → 3 → ... → N',
           ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(f'report/figures/tracking_visualization.{IMG_FORMAT}', format=IMG_FORMAT, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Generated: report/figures/tracking_visualization.{IMG_FORMAT}")


def generate_ablation_chart():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    configs = ['Box\nCrop', '+Motion\nPrediction', '+Pose\nCrop']
    idf1_values = [100.0, 100.0, 100.0]
    id_switches = [1, 2, 0]

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, idf1_values, width, label='IDF1 (%)', color='#2196F3', edgecolor='black')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, id_switches, width, label='ID Switches', color='#F44336', edgecolor='black')

    ax.set_ylabel('IDF1 (%)', color='#2196F3', fontsize=11)
    ax2.set_ylabel('ID Switches', color='#F44336', fontsize=11)
    ax.set_xlabel('Cropping Configuration', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(94, 101)
    ax2.set_ylim(0, 3)

    ax.set_title('Pose Keypoints Ablation Study', fontsize=12, fontweight='bold')

    for bar, val in zip(bars1, idf1_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
               f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2196F3')

    for bar, val in zip(bars2, id_switches):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f'{val}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#F44336')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'report/figures/ablation_study.{IMG_FORMAT}', format=IMG_FORMAT, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Generated: report/figures/ablation_study.{IMG_FORMAT}")


if __name__ == '__main__':
    print("Generating figures for CV MOT Coursework Report...")
    print("=" * 50)

    generate_system_architecture()
    generate_cropping_comparison()
    generate_tracking_comparison()
    generate_tracking_visualization()
    generate_ablation_chart()

    print("=" * 50)
    print("All figures generated successfully!")
    print("Output directory: report/figures/")
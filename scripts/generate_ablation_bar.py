#!/usr/bin/env python3
"""Generate a bar chart for ablation experiment results (subtractive design)."""

import matplotlib.pyplot as plt
import numpy as np

# Data from the updated ablation table (subtractive design)
configs = ['ByteTrack', 'V5', 'V5-Kalman', 'V5-ReID', 'V5-Pose', 'V5-BIoU', 'V5-LowConf']
labels = [
    'ByteTrack\n(开源基线)',
    'V5\n(完整系统)',
    '-Kalman',
    '-ReID',
    '-PoseCrop',
    '-BIoU',
    '-LowConf'
]
idf1 = [68.50, 73.26, 69.14, 64.92, 71.65, 72.18, 72.89]
mota = [50.12, 55.49, 51.87, 48.35, 53.92, 54.65, 55.12]
ids = [315, 236, 278, 325, 251, 244, 240]

x = np.arange(len(configs))
width = 0.28

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')

# IDF1 bars
bars1 = ax1.bar(x - width/2, idf1, width, label='IDF1', color='#DC2626', edgecolor='#991B1B', linewidth=1.2)
bars2 = ax1.bar(x + width/2, mota, width, label='MOTA', color='#2563EB', edgecolor='#1D4ED8', linewidth=1.2)

ax1.set_xlabel('配置', fontsize=11, fontweight='bold')
ax1.set_ylabel('指标 (%)', fontsize=11, fontweight='bold')
ax1.set_title('Wuzhou_MidRoad 消融实验（减法设计）', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.legend(fontsize=10)
ax1.grid(True, axis='y', alpha=0.3)
ax1.spines[['top', 'right']].set_visible(False)
ax1.set_ylim(45, 78)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.6,
             f'{height:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.6,
             f'{height:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

# ID Switches bars
bars3 = ax2.bar(x, ids, width=0.45, color='#059669', edgecolor='#047857', linewidth=1.2)
ax2.set_xlabel('配置', fontsize=11, fontweight='bold')
ax2.set_ylabel('ID Switches', fontsize=11, fontweight='bold')
ax2.set_title('身份切换次数对比', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.grid(True, axis='y', alpha=0.3)
ax2.spines[['top', 'right']].set_visible(False)
ax2.set_ylim(220, 340)

# Add value labels
for bar in bars3:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 3,
             f'{height}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

# Add annotation for key observations
ax1.annotate('ReID贡献最大', xy=(3, 64.92), xytext=(3, 75),
             arrowprops=dict(arrowstyle="->", lw=1.5, color="#DC2626"),
             fontsize=9, fontweight='bold', color='#DC2626')
ax1.annotate('完整系统最优', xy=(1, 73.26), xytext=(1.5, 76),
             arrowprops=dict(arrowstyle="->", lw=1.5, color="#16A34A"),
             fontsize=9, fontweight='bold', color='#16A34A')

fig.suptitle('减法消融实验结果（从完整系统逐个移除模块）', 
             x=0.5, y=0.98, fontsize=13, fontweight='bold', color='#0F172A')
plt.tight_layout()

output_path = 'report/figures/ablation_bar_chart.png'
plt.savefig(output_path, dpi=190, bbox_inches='tight', facecolor='white')
print(f'Generated: {output_path}')
plt.close(fig)

# 面向密集公共场景的 ReID 辅助多目标行人跟踪实验分析

本仓库是计算机视觉课程作业代码包，围绕 **密集公共场景中的多目标行人跟踪（Multi-Object Tracking, MOT）与身份保持问题** 展开。项目重点分析 ReID 外观特征、姿态感知裁剪策略与运动预测对 IDF1、IDP、IDR、MOTA 和 ID Switches 等指标的影响。

> 说明：仓库中不包含真实原始视频、模型权重或隐私数据。为了便于教师快速复现，仓库提供了可在 CPU 上运行的合成 Demo 数据；完整检测/ReID/跟踪流程保留接口结构，可在具备 CUDA、预训练权重和数据后扩展运行。

## 1. 项目结构

```text
CV/
├── README.md
├── requirements.txt              # Demo mode 依赖，CPU 可运行
├── requirements-full.txt         # Full mode 可选依赖
├── generate_figures.py           # 生成报告示意图
├── data/
│   ├── sample_csv/
│   │   ├── sample_gt.csv
│   │   └── sample_tracking_results.csv
│   └── demo_results/
│       ├── baseline_results.csv
│       ├── box_crop_results.csv
│       ├── kalman_results.csv
│       └── pose_crop_results.csv
├── scripts/
│   ├── run_demo.py               # 一键运行 Demo 评估与可视化
│   ├── evaluate.py               # 单文件评估
│   ├── visualize.py              # 跟踪结果可视化
│   └── run_experiments.py        # 多方法对比实验
├── src/
│   ├── config/default.yaml
│   ├── cropping/crop_strategy.py
│   ├── evaluation/metric.py
│   ├── tracking/tracker.py
│   ├── utils/io.py
│   └── visualization/visual.py
├── results/                      # 运行后自动生成指标与可视化结果
└── report/
    └── cv_mot_coursework.tex     # 课程报告 LaTeX 源文件
```

## 2. 快速开始：CPU Demo Mode

### 2.1 安装依赖

```bash
pip install -r requirements.txt
```

### 2.2 一键运行 Demo

```bash
python scripts/run_demo.py
```

该命令会完成：

1. 读取 `data/demo_results/pose_crop_results.csv` 和 `data/sample_csv/sample_gt.csv`；
2. 计算 IDF1、IDP、IDR、MOTA、ID Switches；
3. 生成合成帧上的跟踪可视化，输出到 `results/visualizations/`。

### 2.3 单独运行评估

```bash
python scripts/evaluate.py \
  --pred_file data/sample_csv/sample_tracking_results.csv \
  --gt_file data/sample_csv/sample_gt.csv
```

输出文件默认保存到：

```text
results/metrics/eval_results.txt
```

### 2.4 运行多方法对比实验

```bash
python scripts/run_experiments.py --exp all
```

该命令会比较四组设置：

| 方法 | 含义 |
|---|---|
| Baseline | 基础检测框 + DeepSORT 风格关联 |
| Box Crop | 标准边界框裁剪的 ReID 表征 |
| Kalman | 加入运动预测以平滑轨迹 |
| Pose Crop | 姿态感知裁剪，减少背景和遮挡干扰 |

### 2.5 生成报告插图

```bash
python generate_figures.py
```

输出目录：

```text
report/figures/
```

## 3. 任务定义

给定视频序列中的行人检测结果，多目标跟踪的目标是在连续帧中维持每个行人的身份一致性。密集场景中常见的遮挡、交叉路径、尺度变化和外观相似会导致 ID Switches 增多。本文关注的问题是：

> 在密集公共场景的行人多目标跟踪中，ReID 外观特征与姿态感知裁剪能否提升身份保持能力？

## 4. 方法概要

本项目采用检测驱动跟踪范式：

```text
Video / Frames
    ↓
Person Detection
    ↓
Person Crop / Pose-aware Crop
    ↓
ReID Feature Extraction
    ↓
Motion Prediction + Data Association
    ↓
Tracking Results + Evaluation
```

其中，姿态感知裁剪利用人体关键点估计更稳定的人体区域，减少背景区域和遮挡区域进入 ReID 特征，从而提升身份匹配的稳定性。

## 5. 数据说明

仓库中的 CSV 均为课程 Demo 用合成数据，用于验证代码流程、指标计算和可视化，不包含真实监控视频或隐私信息。

CSV 字段包括：

| 字段 | 说明 |
|---|---|
| `frame` | 帧图像名；Demo 中若没有真实图片，会自动生成合成背景 |
| `idx_frame` | 帧编号 |
| `camera_id`, `camera_name` | 场景/摄像头标识 |
| `box_x1`, `box_y1`, `box_x2`, `box_y2` | 行人检测框坐标 |
| `identity_id` | 评估用身份 ID |
| `primary_uuid`, `secondary_uuid` | 兼容扩展 ID 字段 |
| `xys` | 可选姿态关键点坐标 |

## 6. 评估指标

| 指标 | 含义 | 趋势 |
|---|---|---|
| IDF1 | 身份匹配 F1 分数 | 越高越好 |
| IDP | 身份匹配精确率 | 越高越好 |
| IDR | 身份匹配召回率 | 越高越好 |
| MOTA | 综合 FP、FN、ID Switch 的跟踪准确率 | 越高越好 |
| IDs | ID Switches，身份切换次数 | 越低越好 |

## 7. Full Mode 扩展

完整模式需要额外准备：

- CUDA 环境；
- YOLO/行人检测模型权重；
- ReID 或 CLIP-ReID 特征模型权重；
- 真实或公开 MOT 数据序列。

可选依赖安装：

```bash
pip install -r requirements-full.txt
```

当前仓库优先保证课程评阅时的 **CPU Demo 可复现性**，Full mode 以接口和配置形式保留，便于后续接入真实检测器与 ReID 模型。

## 8. 课程报告对应关系

| 报告部分 | 对应代码/数据 |
|---|---|
| 方法框架 | `src/tracking/`, `src/cropping/`, `src/evaluation/` |
| 实验设置 | `src/config/default.yaml`, `data/demo_results/` |
| 指标计算 | `src/evaluation/metric.py`, `scripts/evaluate.py` |
| 可视化 | `src/visualization/visual.py`, `scripts/visualize.py` |
| 报告插图 | `generate_figures.py`, `report/figures/` |

## 9. 注意事项

1. Demo 数据用于教学复现，不能视为真实场景性能结论。
2. 报告中的方法设计与实验分析强调“可解释的小型研究”，不是完整工业系统复刻。
3. 如果使用真实数据或公开数据集复现实验，应在报告中明确数据来源、预处理方式和划分策略。

## 10. License

本项目仅用于课程作业、教学演示与研究学习。

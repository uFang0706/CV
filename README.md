# 面向密集公共场景的ReID辅助多目标行人跟踪实验分析

## 项目简介

本项目是计算机视觉课程作业，专注于研究密集公共场景下的多目标行人跟踪问题。通过比较不同ReID特征提取方法和姿态感知裁剪策略，分析其对跟踪性能的影响。

## 目录结构

```
cv-mot-coursework/
├── src/
│   ├── config/              # 配置文件
│   │   └── default.yaml
│   ├── cropping/           # 裁剪策略模块
│   │   └── crop_strategy.py
│   ├── evaluation/         # 评估指标计算
│   │   └── metric.py
│   ├── tracking/           # 跟踪器实现
│   │   └── tracker.py
│   ├── utils/               # 工具函数
│   │   └── io.py
│   └── visualization/       # 可视化模块
│       └── visual.py
├── scripts/                 # 运行脚本
│   ├── run_demo.py          # Demo模式运行
│   ├── evaluate.py          # 评估脚本
│   ├── visualize.py         # 可视化脚本
│   └── run_experiments.py   # 实验批量运行
├── data/
│   ├── sample_csv/          # 示例CSV数据
│   └── demo_results/        # Demo结果
├── results/                 # 实验结果输出
│   ├── metrics/
│   └── visualizations/
├── docs/                    # 文档
└── README.md
```

## 运行模式

### Demo模式（推荐先使用）

Demo模式无需CUDA和模型权重，可直接运行评估和可视化功能：

```bash
python scripts/run_demo.py
```

### Full模式

Full模式需要：
- CUDA环境
- YOLO检测模型权重
- ReID特征提取模型权重

```bash
python scripts/run_demo.py --mode full --config src/config/default.yaml
```

## 核心功能

### 1. 评估指标计算

支持MOT Challenge标准指标：
- **IDF1**: 身份识别的F1分数
- **MOTA**: 多目标跟踪准确度
- **ID Switch**: 身份切换次数
- **IDP/IDR**: 身份精确率/召回率

```bash
python scripts/evaluate.py --pred_file data/sample_csv/sample_tracking_results.csv --gt_file data/sample_csv/sample_gt.csv
```

### 2. 结果可视化

```bash
python scripts/visualize.py --csv_file data/sample_csv/sample_tracking_results.csv --output_dir results/visualizations
```

### 3. 批量实验

```bash
python scripts/run_experiments.py --exp all
```

## 实验设计

### 对比方法

1. **Baseline**: YOLO + DeepSORT基础跟踪
2. **Box Crop**: 标准边界框裁剪的ReID特征
3. **Pose Crop**: 基于姿态估计的裁剪策略
4. **Kalman**: Kalman滤波器运动预测加速

### 裁剪策略

#### 标准框裁剪（Box Crop）
- 直接使用检测框进行裁剪
- 简单高效，但可能包含背景噪声

#### 姿态感知裁剪（Pose Crop）
- 利用人体关键点估计
- 减少背景和遮挡区域干扰
- 提取更稳定的人体特征

## 依赖项

```
motmetrics>=1.1.0
numpy>=1.19.0
pandas>=1.0.0
opencv-python>=4.5.0
pyyaml>=5.4.0
prettytable>=0.7.0
```

安装依赖：
```bash
pip install -r requirements.txt
```

## 数据格式

### 输入CSV格式

跟踪结果CSV文件应包含以下列：
- `frame`: 帧图像文件名
- `idx_frame`: 帧编号
- `camera_id`: 摄像头ID
- `camera_name`: 摄像头名称
- `box_x1`, `box_y1`, `box_x2`, `box_y2`: 检测框坐标
- `identity_id`: 行人ID
- `primary_uuid`: 唯一标识符

### 关键点数据格式

可选的xys列存储姿态关键点信息，格式为JSON列表：
```python
[x1, y1, x2, y2, x3, y3, x4, y4]  # 4个关键点坐标
```

## 配置说明

配置文件使用YAML格式，位于 `src/config/default.yaml`：

```yaml
experiment:
  mode: "demo"

detector:
  type: "yolov5"
  conf_thresh: 0.5

tracker:
  type: "deepsort"
  max_iou_distance: 0.7
  max_age: 70

cropping:
  strategy: "box"  # or "pose"

evaluation:
  metrics: ["idf1", "idp", "idr", "mota", "num_switches"]
```

## 注意事项

1. **数据保密**：本项目不包含任何真实监控数据，所有示例数据均为合成数据
2. **敏感信息**：代码和文档中已移除所有与具体场所相关的信息
3. **模型权重**：Full模式需要额外下载模型权重文件

## 实验结果

Demo模式下可获得以下类型的分析结果：

- IDF1/MOTA等指标对比表
- 不同裁剪策略的效果对比
- 跟踪可视化结果

## 扩展方向

1. **更强ReID模型**: 集成更先进的ReID特征提取网络
2. **遮挡感知**: 针对严重遮挡场景的特征更新策略
3. **跨镜跟踪**: 多摄像头场景下的身份匹配

## 课程作业相关

本项目用于以下课程作业：
- 课程名称：计算机视觉
- 作业主题：面向密集公共场景的ReID辅助多目标行人跟踪实验分析
- 核心问题：研究姿态感知裁剪和ReID特征对身份保持能力的影响

## 致谢

本项目基于以下开源工作：
- DeepSORT
- YOLOv5
- CLIP ReID
- MOTMetrics

## 许可证

本项目仅用于课程作业和研究目的。

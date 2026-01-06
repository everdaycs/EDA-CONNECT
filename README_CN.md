# EDA-Connect: 电路原理图识别与网表提取原型系统

本项目是一个基于深度学习和图像处理技术的电路原理图识别原型。它能够自动检测原理图中的电子元器件，并提取它们之间的电气连接关系，最终生成文本格式的网表（Netlist）。

## 项目背景

在电路设计和逆向工程中，将纸质或图片格式的原理图转换为数字化的网表是一项耗时且易出错的工作。本项目旨在通过自动化手段，利用 YOLO 目标检测和形态学图像处理技术，实现从图片到连接关系的自动转换。

## 核心功能

1.  **元器件自动检测**：基于 YOLOv8 模型，识别电阻、电容、芯片、连接器等 60 多种电子元器件。
2.  **导线拓扑提取**：通过自适应阈值和骨架化算法，从复杂的背景中提取电路连线。
3.  **连接关系分析**：自动关联元器件引脚与导线，识别电气网络（Net）。
4.  **可视化验证**：生成带有检测框和导线骨架的叠加图，方便人工校对。
5.  **网表输出**：导出易于阅读的文本格式网表。

## 项目结构

*   `src/circuit_extractor.py`: **核心脚本**。负责执行检测、图像处理和连接图构建。
*   `src/train_yolo.py`: YOLOv8 模型训练脚本。
*   `src/prepare_data.py`: 数据集预处理与划分工具。
*   `src/scan_classes.py`: 标签类别扫描工具。
*   `cpnt_detect_demo/`: 示例数据集（包含图像和 YOLO 格式标注）。
*   `runs/`: 存放训练好的模型权重和训练日志。

## 快速开始

### 1. 安装依赖
确保您的系统中已安装 Python 3.8+，然后运行：
```bash
pip install -r requirements.txt
```

### 2. 训练模型
如果您需要使用自己的数据重新训练模型：
```bash
python src/train_yolo.py
```
训练好的模型将保存在 `runs/detect/train_demo/weights/best.pt`。

### 3. 运行识别与提取
对一张原理图图片进行识别：
```bash
python src/circuit_extractor.py <图片路径>
```
**示例：**
```bash
python src/circuit_extractor.py cpnt_detect_demo/images/001_01_001.png
```

## 技术实现方案

### 第一阶段：元器件识别 (YOLOv8)
系统采用 YOLOv8 目标检测模型。通过对标注好的原理图数据集进行微调（Fine-tuning），模型能够输出每个元器件的类别标签和边界框（Bounding Box）。

### 第二阶段：导线提取 (OpenCV & Scikit-Image)
1.  **预处理**：将图像转为灰度并应用自适应高斯阈值处理，以适应不同的光照条件。
2.  **形态学修复**：使用闭运算和膨胀操作连接细小的断裂，去除孤立噪声点。
3.  **骨架化**：将二值化后的导线缩减为单像素宽度的骨架，保留其拓扑结构。

### 第三阶段：连接图构建 (NetworkX)
1.  **像素图构建**：将骨架像素转换为图节点。
2.  **元件关联**：将位于元器件边界框（及一定扩展区域）内的骨架节点标记为该元件的连接点。
3.  **网络合并**：通过连通分量分析识别独立的电气网络，并对空间邻近的网络进行合并，以增强对断线情况的鲁棒性。

## 输出示例

**output_netlist.txt:**
```text
Components:
Comp_0: Class 1 at [210 433 342 579]
...
Connections:
Net_14: Comp_16 -- Comp_3
Net_16: Comp_1 -- Comp_2
```

## 🧬 弱监督 GNN 提取方案 (gnn_project)

为了解决传统规则算法在应对“断线”、“复杂交叉”和“低像素质量”时的局限性，我们引入了基于**图神经网络 (GNN)** 的提取方案。该方案目前作为实验性原型存储在 `gnn_project/` 目录下。

### 核心原理
1.  **数据流**：利用现有的“元器件检测框”和“导线骨架图”，通过启发式规则自动生成**含噪声的伪标签 (Noisy Pseudo-labels)**。
2.  **模型**：使用基于 PyTorch Geometric 的链路预测模型（如 GraphSAGE 或 GAT），学习两个元器件节点之间是否存在电气连接。
3.  **优势**：GNN 能够学习拓扑模式，具备一定的抗噪性，可填补骨架图中的微小断裂。

### 快速起步 (GNN)

#### 1. 准备数据
将检测框和骨架图转换为图数据（.pt格式）：
```bash
python gnn_project/src/prepare_dataset.py \
  --data_root ./cpnt_detect_demo \
  --output_dir ./gnn_project/processed_data
```

#### 2. 训练链路预测模型
```bash
python gnn_project/src/train.py \
  --data_root ./gnn_project/processed_data \
  --save_dir ./gnn_project/checkpoints \
  --epochs 50
```

#### 3. 执行推理
直接从原理图预测网表：
```bash
python gnn_project/src/infer.py \
  --image ./test.png \
  --det_json ./test_det.json \
  --ckpt ./gnn_project/checkpoints/best_model.pth
```

##两阶段端到端提取 (DeiT + GNN)

本项目新增了基于 Vision Transformer 的两阶段提取方案，鲁棒性更强。

### 1. 结构
- **Stage 1**: 使用 `vit_skel_project` (DeiT) 预测导线骨架图，替代传统的形态学规则。
- **Stage 2**: 使用 `gnn_project` 基于预测的骨架生成连接图，并进行链路预测。

### 2. 也是一键运行
```bash
python pipeline/run_end2end.py \
  --image cpnt_detect_demo/images/001_01_001.png \
  --det_json cpnt_detect_demo/det_json/001_01_001.json \
  --deit_ckpt vit_skel_project/checkpoints/best.pt \
  --gnn_ckpt gnn_project/checkpoints/best_model.pth \
  --out_dir output/001_01_001
```

---

## 未来改进方向
*   **引脚级关键点检测**：目前仅识别元件间的连接，未来可引入关键点检测来精确定位芯片的具体引脚编号。
*   **OCR 增强**：通过文字识别技术读取元件位号（如 R1, C10）和引脚功能。
*   **自训练 (Self-training)**：利用模型的高置信度预测结果反馈优化伪标签。
*   **标准网表支持**：支持导出 SPICE 或 KiCad 等标准的 EDA 网表格式。

## 📁 目录结构

```
EDA-Connect/
├── cpnt_detect_demo/       # 示例数据集 (Images & Labels)
├── pipeline/               # [新增] 端到端运行脚本
├── gnn_project/            # GNN 弱监督网表提取原型
│   ├── src/                # GNN 数据处理、模型与训练代码
│   └── README.md           # GNN 模块独立文档
├── vit_skel_project/       # [新增] DeiT/ViT 骨架提取模型 (Stage 1)
│   └── src/
├── output/                 # 运行结果输出目录
├── runs/                   # YOLO 训练日志与权重
├── src/                    # 源代码
│   ├── circuit_extractor.py # 核心流水线：检测+规则连线
│   ├── train_yolo.py       # YOLO 训练脚本
│   └── ...
├── data.yaml               # YOLO 数据集配置
├── requirements.txt        # 项目依赖
└── README_CN.md            # 项目说明文档
```

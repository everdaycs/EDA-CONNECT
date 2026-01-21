# 电路逻辑连接建模 (Logic Connection Modeling)

本模块旨在通过图神经网络 (GNN) 重建电路图中的电气连接关系 (Netlist)。它作为整个 EDA-Connect 项目的第二阶段，接收第一阶段（元件检测与导线分割）的输出，预测哪些元件端口和导线属于同一个电气网络。

## 1. 核心思想

我们将电路图理解任务建模为**图上的链路预测 (Link Prediction) / 边分类 (Edge Classification)** 问题：
1.  **节点 (Nodes)**: 每一个检测到的元件 (Component) 和每一段导线 (Wire segment/bbox) 作为一个图节点。
2.  **边 (Edges)**: 基于空间邻近性 (距离或 IoU) 构建候选边。
3.  **预测**: GNN 模型学习节点特征和拓扑结构，输出每条候选边的“连通概率”。
4.  **聚类**: 根据预测的连通边，使用连通分量算法将节点划分为若干独立的 Net 簇。

## 2. 项目结构

该模块的代码位于 `gnn_project/` 目录下：

```text
gnn_project/
├── processed_data/      # 存放构建好的 PyG 图数据 (.pt 文件)
├── checkpoints/         # 存放训练好的模型权重
├── src/
│   ├── build_graph.py   # [数据预处理] 将检测结果(JSON)转换为图数据(PyG Data)
│   ├── model.py         # [模型定义] 基于 GAT (Graph Attention Network) 的边分类模型
│   ├── train_gnn.py     # [训练脚本] 训练 GNN 模型并进行验证
│   └── inference.py     # [推理脚本] 对新数据进行预测，输出 Netlist JSON
└── README.md            # 本文档
```

## 3. 详细模块说明

### 3.1 图构建 (`src/build_graph.py`)
负责将图像空间的检测框转换为拓扑图。
- **输入**: 
  - `cpnt_labels/*.json`: 元件检测结果 (BBox, Category)
  - `line_labels/*.json`: 导线检测结果 (Net ID, Segment BBoxes)
- **处理逻辑**:
  - **节点特征**: 归一化的坐标 `(x, y)`、尺寸 `(w, h)`、类型编码 (One-hot: IsWire/IsComp)。
  - **建边策略**: 如果两个节点的空间距离小于阈值 (如 150px) 或存在 IoU 重叠，则建立无向边。
  - **标签生成**: (仅训练模式) 如果两节点在 Ground Truth 中属于同一 Net (基于 ID 或几何相交)，标记边为 Positive (1)，否则为 Negative (0)。
- **输出**: 保存为 PyTorch Geometric `.pt` 文件。

### 3.2 模型架构 (`src/model.py`)
- **Backbone**: 三层 **GAT (Graph Attention Network)**。
  - 利用 Attention 机制聚合邻居信息，捕捉长距离依赖 (通过多层传播)。
- **Classifier**: 一个 MLP (多层感知机)。
  - 输入: 源节点 Embedding拼接 + 目标节点 Embedding拼接 + 边几何特征 (距离, IoU)。
  - 输出: Logits (连接概率)。

### 3.3 训练 (`src/train_gnn.py`)
- **Loss**: `BCEWithLogitsLoss` (带权重的二元交叉熵)，用于处理正负样本不平衡 (负边远多于正边)。
- **Metrics**: 追踪 F1-Score, Accuracy, AUC。
- **Checkpoints**: 保存验证集 F1 分数最高的模型。

### 3.4 推理 (`src/inference.py`)
面向实际应用的接口。
- **输入**: 单张图片的元件 JSON 和导线 JSON。
- **流程**:
  1. 动态构建图结构。
  2. GNN 前向传播得到边概率。
  3. 过滤低置信度边 (Thresholding)。
  4. 使用 `networkx.connected_components` 提取连通子图。
- **输出**: JSON 格式的 Netlist，包含每个 Net 及其包含的元件/导线索引。

## 4. 快速开始

### 环境依赖
需要在 Python 环境中安装 `torch`, `torch_geometric`, `networkx`, `scikit-learn`, `tqdm`。

### 步骤 1: 构建数据集
将你的检测数据 (JSON) 转换为图数据：
```bash
python gnn_project/src/build_graph.py \
  --data_root joint_training_data \
  --save_dir gnn_project/processed_data
```

### 步骤 2: 训练模型
```bash
python gnn_project/src/train_gnn.py
# 模型默认保存至 gnn_project/checkpoints/best_gnn.pth
```
*注: 此脚本中包含硬编码的路径配置，需根据实际情况微调。*

### 步骤 3: 推理 (生成 Netlist)
使用训练好的模型对新数据进行预测：
```bash
python gnn_project/src/inference.py \
  --cpnt_file joint_training_data/cpnt_labels/scenario_35_001_01_001_cpnt.json \
  --wire_file joint_training_data/line_labels/scenario_35_001_01_001_wire_bbox.json \
  --model_path gnn_project/checkpoints/best_gnn.pth
```
输出结果将保存在 `gnn_project/` 目录下，文件名为 `inference_output_{basename}.json`。

## 5. 数据格式示例

### 推理输出 (Inference Output JSON)
```json
{
    "NET_001": [
        {
            "type": "component",
            "bbox": [1027, 548, 1043, 578],
            "original_index": 0
        },
        {
            "type": "wire",
            "bbox": [1030, 550, 1040, 560],
            "original_index": 75
        }
    ],
    "NET_002": [ ... ]
}
```
该结构直接对应电路中的电气网络，可进一步用于 SPICE 网表转换。

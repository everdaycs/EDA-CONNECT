# 元件逻辑连接重建方案 (Route B: Component-only Connectivity)

本方案旨在**跳过高难度的细小导线检测（Wire Detection）训练**，直接通过“元件-元件”之间的视觉特征来推断电气连接关系。系统利用现有的元件检测结果，配合弱监督的分类模型，重建电路的逻辑网表。

## 1. 核心思想

将电路连接重建任务简化为**边分类（Edge Classification）**问题：
*   **输入**: 两个元件的检测框 (BBox)。
*   **证据**: 两个元件之间的图像区域（Corridor Patch）是否包含连接线索。
*   **输出**: 二分类概率（连通 / 不连通）。

该方法直接关注“连接关系”而非“导线像素位置”，鲁棒性更高。

## 2. 系统 Pipeline

### Stage 1: 候选边生成
为了避免 $O(N^2)$ 的全图计算，系统对每个元件使用 **k-近邻 (kNN)** 算法（默认 $k=12$）生成空间上邻近的候选连接对，并设置最大距离阈值过滤无效边。

### Stage 2: 特征提取
对每一条候选边 $(u, v)$，提取多模态特征：
1.  **视觉特征 (Corridor Patch)**: 裁剪覆盖两个元件及其连线区域的图像 Patch。
    *   处理：Resize 到 $128 \times 64$，并进行随机旋转、噪声增强。
2.  **几何特征 (Geometric Vector)**: 提取 8 维向量，包含相对距离、方向角度 ($dx, dy$)、以及两个 BBox 的宽高。

### Stage 3: 连接判别模型 (EdgeCNN)
*   **Backbone**: 使用轻量级 **ResNet18** 提取 Corridor Patch 的视觉 Embedding。
*   **Fusion**: 将视觉 Embedding 与几何特征拼接。
*   **Head**: 通过 MLP 输出连接概率 $P(connect|u, v)$。

### Stage 4: 逻辑聚类
根据模型输出的概率矩阵，建立加权无向图，设置阈值（如 0.5）后使用 **并查集/连通分量算法** 将元件聚类。每个连通簇对应一个电路网络 (Net ID)。

---

## 3. 完整训练流程 (Detailed Training Workflow)

本方案采用**弱监督学习（Weakly Supervised Learning）**。训练的核心在于：不需要人工手动标注元件间的“边”，而是通过现有的导线 NetID 标注自动推导出元件间的连接关系，并以此训练一个端到端的分类器。

### 3.1 环境配置与依赖
确保 Python 环境中已安装以下库：
*   **基础**: `torch`, `torchvision`, `numpy`, `opencv-python`, `PyYAML`
*   **图像增强**: `albumentations` (关键：用于 Corridor Patch 的物理变换)
*   **图处理**: `networkx` (用于连通分量聚类)
*   **几何计算**: `scipy` (用于 kNN 候选边生成)

### 3.2 目录与数据组织
训练前必须按照以下结构准备数据（通常位于 `joint_training_data/`）：
```text
/data_root/
  ├── images/            # 电路图原始文件 (.png / .jpg)
  ├── cpnt_labels/       # 元件检测 Ground Truth (JSON，含 bbox)
  └── line_labels/       # 导线标注 GT (JSON，按 NetID 分组，由第一阶段生成)
```

### 3.3 训练全流程步骤

#### 第一步：弱标签自动生成 (Offline Weak Labeling)
执行 `train_edge.py` 时，系统会自动调用 `weak_labeling.py`。
1.  **引脚/边缘判定**: 算法在元件 BBox 周围建立宽 4px 的“判定环”（Ring）。
2.  **导线碰撞检测**: 将导线标注渲染为 Mask 并进行膨胀。若导线 Mask 与元件环的重叠面积达到阈值，则认为该元件连接到该 NetID。
3.  **连接对推导**: 属于同一个 NetID 的两个元件被标记为**正样本 ($y=1$)**。如果不属于任何共同网络但在空间上邻近，则通过 kNN 被选为候选边并标记为**负样本 ($y=0$)**。

#### 第二步：难负样本均衡 (Hard Negative Mining)
由于图中绝大多数元件对是不相连的，系统采用 **1:3 的正负样本比**。负样本优先挑选空间距离近的边，这些边在视觉上最具有误导性（例如两根互不相连的导线靠得很近），是模型训练的关键。

#### 第三步：视觉-几何融合训练 (Training)
1.  **输入**: 
    *   **Corridor Patch**: 包含两个元件间可能存在导线的图像切块。
    *   **Geometry Tensor**: 包含 $dx, dy, distance, \theta$ 等几何先验知识。
2.  **Loss 函数**: 使用 `BCEWithLogitsLoss`。为应对样本不平衡，我们在 Loss 中为正样本设置了较高的权重 (`pos_weight=2.0`)。
3.  **优化器**: 使用 `AdamW`，初始学习率 $10^{-4}$，并自动保存验证集准确率最高的权重。

### 3.4 启动命令
运行以下命令即可：
```bash
# 确保在项目根目录
python frcnn_project/src/connectivity/train_edge.py
```

### 3.5 训练产物
*   **Cache**: `frcnn_project/outputs/edge_cache/train_edges.pt` (保存了生成的弱标签，下次训练可快速加载)。
*   **Checkpoint**: `frcnn_project/outputs/checkpoints_edge/best_edge_model.pth` (最终推理所需的权重文件)。

---

## 4. 推理与评估 (Inference & Evaluation)

为了隔离元件检测器的误差（漏检/误检）并专注于连接预测算法本身，我们支持直接使用 **Ground Truth (GT)** 元件标注进行评估。

### 4.1 使用 GT 元件推理
您可以直接将推理脚本的 `--cpnt` 参数设置为数据集中的标注 JSON 文件：
```bash
python frcnn_project/src/connectivity/infer_connectivity.py \
  --img joint_training_data/images/001.png \
  --cpnt joint_training_data/cpnt_labels/001_cpnt.json \
  --model frcnn_project/outputs/checkpoints_edge/best_edge_model.pth
```

### 4.2 独立评估脚本 (`eval_connectivity.py`)
我们提供了一个自动评估工具，可以遍历整个数据集，使用 GT 元件计算网表恢复的准确率：
```bash
python frcnn_project/src/connectivity/eval_connectivity.py
```
**评估指标解释：**
- **Pairwise Accuracy**: 元件对连接状态预测的总体准度。
- **Pairwise Recall**: 衡量“应连接”被成功识别的比例（防止断路）。
- **Pairwise Precision**: 衡量“不应连接”被正确区分的比例（防止短路）。

### 4.3 结果可视化 (`visualize_connectivity.py`)
为了直观查看模型预测的连接关系，可以使用可视化脚本：
```bash
python frcnn_project/src/connectivity/visualize_connectivity.py \
  --img joint_training_data/images/001.png \
  --cpnt joint_training_data/cpnt_labels/001_cpnt.json \
  --out frcnn_project/outputs/vis/demo.png
```
该脚本会将属于同一个 Net 的元件用**相同颜色**框出，并在它们之间绘制**连接线**。

### 4.4 输出格式 (JSON)
直接输出重建后的 Netlist：
```json
[
  {
    "net_id": "NET_000",
    "component_ids": [0, 5, 12],
    "components": [
       {"class_id": 3, "bbox": [...]},
       ...
    ]
  },
  ...
]
```

---

## 5. 项目文件结构

代码位于 `frcnn_project/src/connectivity/`：

| 文件 | 说明 |
| :--- | :--- |
| `weak_labeling.py` | **核心**: 离线脚本，从 GT 导线推导元件连接关系，生成训练标签。 |
| `edge_dataset.py` | PyTorch Dataset 定义，包含 Corridor Patch 的裁剪与 Albumentations 增强。 |
| `edge_model.py` | 定义 EdgeCNN 模型结构 (ResNet18 + Geometry Fusion)。 |
| `eval_connectivity.py`| **评估**: 基于 GT 元件评估连接准确率的脚本。 |
| `visualize_connectivity.py`| **可视化**: 将预测的连接关系绘制在原图上。 |
| `train_edge.py` | 训练入口脚本。 |
| `infer_connectivity.py` | 推理入口脚本，包含 kNN 建图、模型预测与连通分量聚类。 |
| `README.md` | 本说明文档。 |

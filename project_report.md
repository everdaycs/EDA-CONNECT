# AI 辅助原理图识别与网表提取系统开发进展报告

## 1. 项目概述 (Overview)
本项目旨在开发一套基于深度学习的端到端系统，能够从电路原理图（Schematic Images）中自动识别元器件与导线，推理电气连接关系，并最终输出标准的网表（Netlist）。系统采用“目标检测 + 语义分割 + 图神经网络”的混合架构，克服了传统视觉算法在复杂噪声、断线和非曼哈顿布局下的局限性。

## 2. 核心技术架构 (Technical Architecture)
本项目构建了 Two-Stage（两阶段）深度学习流水线，辅以自动化数据工程工具。

### 2.1 阶段一：元器件感知与导线提取 (Perception Stage)
*   **元器件检测 (Component Detection)**：
    *   **模型**：采用 State-of-the-Art 的 **YOLOv11x** 目标检测模型。
    *   **能力**：支持电阻、电容、IC 等多种元器件的定位与分类。
    *   **现状**：已集成 Ground Truth (GT) 标注支持，并完成了 YOLO 标签到像素级坐标 (JSON) 的自动化转换工具。
*   **导线骨架提取 (Skeleton Extraction)**：
    *   **方案 A (Deep Learning)**：基于 **DeiT (Data-efficient Image Transformer)** 的语义分割模型。相比 CNN，Transformer 能更好地捕捉全局上下文，有效抵抗原理图中的文字遮挡和阴影噪声。
    *   **方案 B (Rule-based Baseline)**：基于 OpenCV 的形态学骨架提取算法（自适应阈值 + 连通域分析），作为弱监督学习的 Pseudo-label 生成器和备份方案。
    *   **现状**：双方案均已实装，可通过参数无缝切换。

### 2.2 阶段二：拓扑推理与重构 (Reasoning Stage)
*   **图构建 (Graph Construction)**：
    *   将像素级的导线骨架抽象为图（Graph）结构，元器件作为节点（Node），导线路径作为边（Edge）。
    *   引入空间几何特征（IoU、距离场）来处理元器件引脚与导线的物理挂载。
*   **连接预测 (Link Prediction)**：
    *   **模型**：**GraphSAGE (Graph Neural Network)**。
    *   **逻辑**：模型学习电路拓扑的概率分布，能智能推断视觉上断裂但逻辑上连通的线路（如被文字切断的线），输出鲁棒的连接概率。
*   **网表生成 (Netlist Generation)**：
    *   基于 GNN 推理结果，通过连通分量合并算法（Super-net merging）生成最终的电气连接表，输出格式可兼容 EDA 软件。

## 3. 当前开发进度 (Current Status)

| 模块 | 功能描述 | 状态 | 备注 |
| :--- | :--- | :--- | :--- |
| **Data Engine** | YOLO 标签转 JSON / 多进程数据预处理 | ✅ 完成 | 支持 500+ 数据集秒级转换 |
| **Pipeline** | `run_end2end.py` 一键式推理脚本 | ✅ 完成 | 集成全流程，支持 GT 注入 |
| **Stage 1 (ViT)** | DeiT 骨架分割模型 | ✅ *部分完成* | 模型已跑通，需进一步从头训练以提升精度 |
| **Stage 2 (GNN)** | 图神经网络连接推理 | ✅ 完成 | 已实现从骨架到网表的完整映射 |
| **Visualization** | 结果叠加显示 (Overlay) | ✅ 完成 | 提供红线绿框的直观验证图 |

## 4. 演示验证 (Demo Validation)
目前的 `v0.1` 原型已在演示数据集上跑通全流程：
*   **输入**：一张原始 PNG 原理图 + 标注文件。
*   **执行**：单命令 `python pipeline/run_end2end.py`。
*   **输出**：
    1.  `skeleton_deit.png`: 干净的导线层。
    2.  `output_netlist.txt`: 包含 `Net_X: Comp_A -- Comp_B` 的文本化网表。
    3.  `overlay.png`: 视觉确认图。

## 5. 下一步计划 (Next Steps)
1.  **Stage 1 模型迭代**：在全量 514 张数据集上训练 DeiT 模型，替代目前的预训练权重，提升对细微导线的捕捉能力。
2.  **OCR 集成**：引入文字识别模块，读取元器件旁的 Reference Designator（如 "R1", "C12"），替代目前的 "Comp_ID"，使网表更具工程实用价值。
3.  **端到端微调**：联合训练 YOLO 与 GNN，进一步降低级联误差。

---
*报告生成时间：2026年1月*

# Technical Report: Deep Learning Based Schematic Netlist Extraction v0.1

## 1. Executive Summary
This project implements a **Two-Stage Deep Learning Pipeline** designed to digitize electronic schematic images into SPICE-compatible netlists. Unlike traditional heuristic approaches (e.g., probabilistic Hough Transform), our system leverages a hybrid neuro-symbolic architecture combining **Vision Transformers (ViT)** for semantic segmentation and **Graph Neural Networks (GNN)** for topological reasoning. This ensures high robustness against image noise, text occlusion, and non-Manhattan routing artifacts.

## 2. System Architecture & Algorithms

### 2.1 Stage I: Visual Perception (Pixel-Level)
The perception module is responsible for extracting structured geometric primitives from unstructured raster images.

*   **Object Detection (YOLOv11x)**
    *   **Architecture**: Ultralytics YOLOv11x (Extra Large), optimized for small object detection.
    *   **Inference**: Single-shot detection of discrete components (Resistors, Capacitors, ICs).
    *   **Post-Processing**: Coordinate denormalization converting relative YOLO formats `(cx, cy, w, h)` to absolute pixel-space Bounding Boxes for graph node mapping.

*   **Skeleton Extraction (Hybrid Approach)**
    *   **Deep Learning Path (ViT/DeiT)**:
        *   **Backbone**: `deit_tiny_patch16_224` (Data-efficient Image Transformer).
        *   **Decoder**: Custom `ConvTranspose2d` upsampling head to restore spatial resolution from $14 \times 14$ feature maps to $H \times W$ segmentation masks.
        *   **Loss Function**: Hybrid loss $\mathcal{L}_{total} = \mathcal{L}_{Dice} + \lambda \mathcal{L}_{BCE}$ to handle class imbalance (thin wires vs. large background).
    *   **Morphological Fallback (CV)**:
        *   Adaptive Gaussian Thresholding ($\sigma=11, C=2$).
        *   Morphological Closing/Opening for noise reduction.
        *   Iterative Skeletonization (Zhang-Suen algorithm implementation via `skimage`).

### 2.2 Stage II: Topological Reasoning (Graph-Level)
The reasoning module converts pixel-space primitives into a logical connectivity graph.

*   **Graph Construction**:
    *   **Nodes $V$**: Constructed from Component Centroids and Skeleton Junction points.
    *   **Edges $E$**: Established via physical adjacency (8-neighborhood) in the skeleton map and ROI (Region of Interest) inclusion tests ($\text{dist}(p, \text{BBox}) < \delta$).
    *   **Features $X$**: Normalized coordinates $(x, y)$, component class logits, and geometric embeddings.

*   **Graph Neural Network (GraphSAGE)**:
    *   **Model**: GraphSAGE (Graph Sample and Aggregate) specialized for inductive representation learning on large graphs.
    *   **Mechanism**: Aggregates features from local neighborhoods to generate node embeddings $h_v$.
    *   **Edge Prediction**: A binary classifier MLP operating on concatenated node embeddings $MLP(h_u || h_v)$ to predict electrical connectivity probability $P(u \sim v)$.
    *   **Advantage**: Capable of inferring "logical connectivity" even when visual connectivity is broken by text labels or noise.

## 3. Implementation Roadmap & Status

### 3.1 Engineering Stack
*   **Frameworks**: PyTorch, PyTorch Geometric (PyG), Timm (PyTorch Image Models), Ultralytics.
*   **Pipeline Orchestration**: Python-based `subprocess` control with environment isolation.
*   **Optimization**: `ProcessPoolExecutor` for parallel graph generation; Vectorized NumPy operations for distance matrix calculations.

### 3.2 Module Status Matrix

| Module | Component | Tech Stack | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **P-1** | Component Detector | YOLOv11x | ✅ Validated | Integrated with GT injection via `--label` |
| **P-2** | Skeleton Segmenter | DeiT-Tiny + Decoder | ⚠️ Prototype | Architecture defined; Training required on full dataset |
| **P-2** | CV Extractor | OpenCV / Morph | ✅ Completed | Robust baseline implemented in `skeleton_cv.py` |
| **R-1** | Graph Builder | NetworkX / Numpy | ✅ Optimized | Refactored for modularity and speed |
| **R-2** | Connectivity Engine | GraphSAGE | ✅ Validated | E2E Inference logic verified in `gnn_project` |

## 4. Current Workflow (v0.1)
The system currently operates via the unified entry point `pipeline/run_end2end.py`:
1.  **Injection**: Accepts Raster Image + (Optional) GT Labels.
2.  **Perception**: Inferences Component BBoxes and Wire Skeletons.
3.  **Graph Transformation**: Maps BBoxes to graph nodes; vectorized mapping of skeleton pixels to edges.
4.  **Reasoning**: GNN predicts missing links and aggregates connected components.
5.  **Netlist Compilation**: Serializes graph components into textual Netlist format.

## 5. Future Technical Directions
1.  **Optical Character Recognition (OCR)**: Integration of PaddleOCR/Tesseract to extract Reference Designators (e.g., "R1", "U2") for netlist semantic enrichment.
2.  **End-to-End Fine-tuning**: Implementation of a Differentiable Rasterization layer to allow backpropagation from the Netlist loss back to the Skeleton extraction network.
3.  **Graph Attention Mechanisms**: Upgrade from GraphSAGE to GAT (Graph Attention Networks) to better handle complex crossing wires (Junction vs. Crossover).

# Faster R-CNN Project for EDA-Connect

This sub-project implements a robust Faster R-CNN pipeline for detecting electronic components on schematics.

## Directory Structure
- `configs/` : Configuration files (parameters, paths).
- `src/` : Source code for model, training, inference.
- `tools/` : Utilities for data sanity checks and conversion.
- `outputs/` : Training logs, model checkpoints.

## 1. Installation
Ensure you are in the `frcnn_project` folder.

```bash
pip install -r requirements.txt
```
*Note: PyTorch for Mac M1/M2 (MPS) is supported automatically.*

## 2. Data Preparation
Run the sanity check tool. This will:
1. Validate all YOLO label files.
2. Filter out invalid/empty boxes.
3. Generate `data_split.json` containing Train/Val splits.

```bash
# Example assuming data is in ../cpnt_detect_demo
python tools/sanity_check.py --data_root ../cpnt_detect_demo --img_dir_name images --lbl_dir_name labels --output data_split.json
```

## 3. Training
Train the model using the default configuration (optimized for Recall).

```bash
# Run training (auto-detects MPS/CUDA)
python src/train.py --config configs/default.yaml
```
Output will be in `outputs/checkpoints/` (contains `best_model.pth` and `last.pth`).

## 4. Evaluation
Evaluate the model on the Validation set defined in `data_split.json`.

```bash
# This is automatically called during training, but to run standalone:
python src/eval.py --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pth
```
*(Note: You may need to create a dedicated eval script wrapper or just modify train.py to support eval-only mode, currently eval is embedded in train loop or use inference on val set)*.

## 5. Inference
Run inference on a folder of images.

```bash
python src/infer.py \
  --input ../cpnt_detect_demo/images \
  --checkpoint outputs/checkpoints/best_model.pth \
  --output final_results.json \
  --vis_dir outputs/visualization \
  --conf_thresh 0.3
```

## Key Configuration for High Recall (configs/default.yaml)
- **Anchors**: Added small sizes `[8, 16]` and slender aspect ratios `[0.2, 5.0]`.
- **RPN**: `nms_thresh` raised to `0.85` to keep more proposals. `post_nms_top_n` increased to `2000` (training).
- **Inference**: `box_score_thresh` set low (`0.05`).

## Tiling Support
Tiling is configured in `configs/default.yaml` under `inference.tiling`. Logic needs to be fully enabled in `src/infer.py` if dealing with 4K+ resolution images (current `infer.py` uses simple resize for v0.1).

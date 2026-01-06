# Schematic Netlist Extraction with Weakly Supervised GNN

This project implements a GNN-based pipeline to extract netlists from schematic images using noisy pseudo-labels derived from heuristics.

## Directory Structure
```
src/
  prepare_dataset.py  # Data preprocessing & Pseudo-label generation
  models.py           # PyG GNN Model definition
  train.py            # Training loop
  infer.py            # Inference pipeline
requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
# Ensure you have torch-geometric installed compatible with your CUDA version
```

## 1. Prepare Data
Expects data organized as:
- `data_root/images/*.png`
- `data_root/det/*.json` (YOLO output: [{"class_id": int, "bbox": [x1,y1,x2,y2]}])
- `data_root/skel/*.png` (Wire skeleton binaries)

Run:
```bash
python src/prepare_dataset.py \
  --data_root ./my_data \
  --output_dir ./processed_data \
  --r_graph 300
```
This generates `.pt` graph files with weak supervision labels.

## 2. Train GNN
```bash
python src/train.py \
  --data_root ./processed_data \
  --save_dir ./checkpoints \
  --epochs 50 \
  --batch_size 16 \
  --model_type SAGE
```

## 3. Inference
```bash
python src/infer.py \
  --image ./test_img.png \
  --det_json ./test_det.json \
  --ckpt ./checkpoints/best_model.pth \
  --out_dir ./results
```

## Top 5 "Gotchas" & Troubleshooting

1. **Skeleton Quality**: If `skel/*.png` has too many breaks, the heuristic `prepare_dataset.py` will generate broken Nets (false negatives). 
   - *Fix*: Tune `d_attach` threshold or improve skeletonization morphology (dilation before thinning).
2. **Graph Connectivity**: `r_graph` is critical. If too small, distant components on a long wire will never have an edge to classify.
   - *Fix*: Visualize the initial graph coverage. Consider KNN (k=10) instead of Radius if component density varies wildly.
3. **Class Imbalance**: True connections are < 1% of all pairwise edges.
   - *Fix*: The `pos_weight` in `train.py` is hardcoded to 5.0. calculate `num_neg/num_pos` dynamically if recall is low.
4. **Coordinate Normalization**: `prepare_dataset.py` uses placeholder `W,H=640`.
   - *Fix*: Ensure you normalize by the **actual** image dimensions, otherwise geometric embeddings will be nonsense.
5. **Overfitting to Heuristics**: The GNN might just learn to mime the skeleton errors.
   - *Fix*: Use early stopping. The validation set is also noisy, so manual inspection of `infer_overlay.png` is better than trusting `val_f1` blindly.

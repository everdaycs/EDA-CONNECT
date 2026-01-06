# ViT-based Skeleton Extraction Project

This module implements **Stage 1** of the 2-stage EDA-Connect pipeline:
Using a Vision Transformer (DeiT/ViT) to predict the wire skeleton mask from a schematic image.

## Structure
- `src/model.py`: ViT-based segmentation model (Encoder-Decoder)
- `src/train.py`: Training script
- `src/infer.py`: Inference script
- `src/gen_rules.py`: Utility to generate rule-based ground truth masks for training

## Usage

### 1. Generate Pseudo-Labels (Rule-based)
Before training, generate standard masks from your images:
```bash
python vit_skel_project/src/gen_rules.py \
  --img_dir ../cpnt_detect_demo/images \
  --out_dir ../cpnt_detect_demo/skeleton_rule
```

### 2. Train DeiT Skeleton Model
```bash
python vit_skel_project/src/train.py \
  --img_dir ../cpnt_detect_demo/images \
  --skel_dir ../cpnt_detect_demo/skeleton_rule \
  --save_dir vit_skel_project/checkpoints \
  --epochs 50 --batch 8
```

### 3. Inference
```bash
python vit_skel_project/src/infer.py \
  --img_path ../cpnt_detect_demo/images/test.png \
  --ckpt vit_skel_project/checkpoints/best.pt \
  --out_mask output/test/skeleton_deit.png
```

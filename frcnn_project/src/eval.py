import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

def evaluate_coco(model, data_loader, device):
    model.eval()
    results = []
    
    # We need to construct GT similarly for pycocotools if it's not already
    # But PyTorch dataloader yields batches.
    # To use COCOeval, we construct a COCO object for GT and one for DT.
    
    gt_annotations = []
    images_info = []
    
    # We can rebuild GT from the split file directly instead of iterating dataloader for GT
    # because iterating dataloader is slow.
    # However, keeping it simple: Iterate dataloader to synchronize predictions.
    
    # Proper way:
    # 1. Iterate val set -> collect DT (detection results)
    # 2. Build GT json from dataset.annotations
    
    # Since SchematicDataset has access to the split file, we can just load it.
    ds = data_loader.dataset
    
    # Build GT Dictionary for COCO
    coco_gt_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": str(i)} for i in range(1, 1000)] # Dummy categories
    }
    
    # Fill GT from dataset
    # Note: dataset.annotations has all images, we only need 'val' split
    # ds.img_paths contains the val images.
    
    img_id_map = {} # path -> id
    ann_id_cnt = 1
    
    for idx, img_path in enumerate(ds.img_paths):
        # We need H, W. Assuming we read them or stored them?
        # Dataset doesn't store H,W in annotations cache for simplicity.
        # But we need image_id.
        img_id = idx + 1
        img_id_map[img_path] = img_id
        
        # We dummy H,W here because COCO eval doesn't strictly need them for bbox eval
        coco_gt_dict["images"].append({"id": img_id, "file_name": os.path.basename(img_path), "height": 1024, "width": 1024})
        
        anns = ds.annotations[img_path]
        for ann in anns:
            x1, y1, x2, y2 = ann['bbox']
            w = x2 - x1
            h = y2 - y1
            coco_gt_dict["annotations"].append({
                "id": ann_id_cnt,
                "image_id": img_id,
                "category_id": ann['class_id'] + 1, # +1 shift
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id_cnt += 1
            
    coco_gt = COCO()
    # Suppress print
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()
    
    # Collect Predictions
    results = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            # targets is tuple of dicts
            # We need image_ids to map back
            # SchematicDataset returns image_id tensor
            
            for i, output in enumerate(outputs):
                img_id = targets[i]['image_id'].item()
                # If using batched loader, targets['image_id'] needs care
                # But here targets is list of dicts.
                
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                for b, s, l in zip(boxes, scores, labels):
                    # COCO bbox is xywh
                    x1, y1, x2, y2 = b
                    results.append({
                        "image_id": img_id + 1, # 1-based index we created above corresponds to idx + 1?
                        # wait, collate_fn preserves order? Dataloader shuffle=False.
                        # dataset index 0 -> img ID 1.
                        # target['image_id'] from __getitem__ is 'idx'. 
                        # So target['image_id'] is 0-based index. 
                        # Our map is idx+1. So use target['image_id'] + 1.
                        "category_id": int(l),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(s)
                    })
    
    if not results:
        print("No detections!")
        return {'map': 0.0, 'ar_max': 0.0}
        
    coco_dt = coco_gt.loadRes(results)
    
    # Evaluate
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Recal optimized evaluation
    coco_eval.params.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract Metrics
    stats = coco_eval.stats
    # stats[0] = mAP 0.5:0.95
    # stats[1] = mAP 0.5
    # stats[8] = AR maxDets=100
    
    # Custom Recall Per Class
    # We can dig into coco_eval.eval['precision'] or similar, but simplified here:
    # Just return main metrics
    
    return {'map': stats[0], 'ar_max': stats[8]}

if __name__ == "__main__":
    import argparse
    import yaml
    from src.dataset import SchematicDataset, collate_fn
    from torch.utils.data import DataLoader
    from src.model import get_model
    from src.utils.general import get_device

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='val')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg['train']['device'])
    
    # Dataset
    data_split = cfg['data']['split_file']
    if not os.path.exists(data_split):
        print(f"Error: Split file {data_split} not found. Run sanity_check.py first.")
        exit(1)
        
    dataset = SchematicDataset(data_split, split=args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, 
                        num_workers=cfg['train']['num_workers'], collate_fn=collate_fn)

    # Model
    with open(data_split) as f:
        meta = json.load(f)
        num_classes = meta['num_classes'] + 1

    model = get_model(num_classes, cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    
    model.to(device)
    
    print(f"Evaluate on {len(dataset)} images...")
    metrics = evaluate_coco(model, loader, device)
    print(f"\nFinal Results:\n mAP (0.5:0.95): {metrics['map']:.4f}\n Recall (AR max=100): {metrics['ar_max']:.4f}")

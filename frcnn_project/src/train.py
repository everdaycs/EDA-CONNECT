import os
import yaml
import torch
import math
import sys
import argparse
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import SchematicDataset, collate_fn
from src.model import get_model
from src.utils.general import get_device, AverageMeter, seed_everything
from src.eval import evaluate_coco

def train(args):
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    seed_everything(cfg['train']['seed'])
    device = get_device(cfg['train']['device'])
    print(f"Using device: {device}")
    
    # Dataset
    dataset_train = SchematicDataset(cfg['data']['split_file'], split='train')
    dataset_val = SchematicDataset(cfg['data']['split_file'], split='val')
    
    loader_train = DataLoader(dataset_train, batch_size=cfg['train']['batch_size'], 
                              shuffle=True, num_workers=cfg['train']['num_workers'], 
                              collate_fn=collate_fn)
                              
    loader_val = DataLoader(dataset_val, batch_size=1, # Eval batch size usually 1
                            shuffle=False, num_workers=cfg['train']['num_workers'], 
                            collate_fn=collate_fn)
                            
    # Model
    # Load num_classes from split file metadata
    with open(cfg['data']['split_file']) as f:
        meta = json.load(f)
        num_classes = meta['num_classes'] + 1 # +1 for background
        
    model = get_model(num_classes, cfg)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg['train']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr=cfg['train']['lr'], 
                                    momentum=cfg['train']['momentum'], 
                                    weight_decay=cfg['train']['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(params, lr=cfg['train']['lr'], 
                                      weight_decay=cfg['train']['weight_decay'])
                                      
    # Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                   step_size=cfg['train']['lr_step_size'], 
                                                   gamma=cfg['train']['lr_gamma'])
                                                   
    # Resume
    start_epoch = 0
    best_mAP = 0.0
    checkpoint_dir = os.path.join(cfg['data']['output_dir'], 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train Loop
    num_epochs = cfg['train']['epochs']
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        loss_meter = AverageMeter()
        
        pbar = tqdm(loader_train, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, targets in pbar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            loss_meter.update(losses.item())
            
            # Print detailed losses occasionally
            if loss_meter.count % cfg['train']['print_freq'] == 0:
                pbar.set_postfix({'loss': f"{loss_meter.avg:.4f}", 
                                  'cls': f"{loss_dict['loss_classifier'].item():.3f}",
                                  'box': f"{loss_dict['loss_box_reg'].item():.3f}"})
                                  
        lr_scheduler.step()
        
        # Save Last
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_mAP': best_mAP
        }, os.path.join(checkpoint_dir, 'last.pth'))
        
        # Evaluation
        if (epoch + 1) % cfg['train']['eval_interval'] == 0:
            print("Running evaluation...")
            metrics = evaluate_coco(model, loader_val, device)
            mAP = metrics['map'] # mAP 0.5:0.95
            ar_max = metrics['ar_max']
            
            print(f"Epoch {epoch+1}: mAP={mAP:.4f}, AR={ar_max:.4f}")
            
            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
                print("New best model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    train(args)

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import glob

from weak_labeling import WeakLabeler
from edge_dataset import EdgeDataset
from edge_model import EdgeCNN

def prepare_data(config):
    """
    Offline step: Generate edge labels from images + cpnt + wire GT
    And save to a cache file (list of dicts).
    """
    cache_dir = config['data']['edge_cache_dir']
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'train_edges.pt')
    
    if os.path.exists(cache_path):
        print(f"Loading cached edges from {cache_path}")
        return cache_path

    labeler = WeakLabeler(config)
    all_edges = []
    
    # Iterate over images in the joint folder
    # Assuming match by filename stem
    img_dir = config['data']['image_dir']
    cpnt_dir = config['data']['cpnt_label_dir']
    wire_dir = config['data']['wire_label_dir']
    
    images = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))
    
    print("Generating weak labels...")
    for img_p in tqdm(images):
        base = os.path.splitext(os.path.basename(img_p))[0]
        
        # Check matching labels
        c_p = os.path.join(cpnt_dir, base + "_cpnt.json")
        w_p = os.path.join(wire_dir, base + "_wire_bbox.json")
        
        if not os.path.exists(c_p) or not os.path.exists(w_p):
            continue
            
        edges = labeler.generate_labels(img_p, c_p, w_p)
        
        # HARD NEGATIVE MINING (Simplified)
        # Keep all positives, subsample negatives
        positives = [e for e in edges if e['label'] == 1]
        negatives = [e for e in edges if e['label'] == 0]
        
        n_pos = len(positives)
        if n_pos > 0:
            ratio = config['data']['edge_gen']['negative_ratio']
            n_neg = min(len(negatives), n_pos * ratio)
            
            # Random subsample or based on distance (already effectively distance based by KNN)
            import random
            random.shuffle(negatives)
            subset_neg = negatives[:n_neg]
            
            epoch_edges = positives + subset_neg
            
            # Inject img_name
            img_name = os.path.basename(img_p)
            for e in epoch_edges:
                e['img_name'] = img_name
                
            all_edges.extend(epoch_edges)
            
    print(f"Total edges generated: {len(all_edges)}")
    torch.save(all_edges, cache_path)
    return cache_path

def train():
    # Load Config
    config_path = "/home/kaga/Desktop/EDA-Connect/frcnn_project/configs/edge/edge_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # 1. Prepare Data
    cache_path = prepare_data(config)
    
    # 2. Dataset
    dataset = EdgeDataset(cache_path, config['data']['image_dir'])
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'], shuffle=False, num_workers=4)
    
    # 3. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeCNN().to(device)
    
    # 4. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['train']['pos_weight']]).to(device))
    
    # 5. Loop
    epochs = config['train']['epochs']
    best_acc = 0.0
    
    os.makedirs(config['train']['save_dir'], exist_ok=True)
    
    for ep in range(epochs):
        model.train()
        loss_sum = 0
        
        for batch in tqdm(train_loader, desc=f"Ep {ep+1}"):
            patches, geoms, labels = batch
            patches = patches.to(device)
            geoms = geoms.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            logits = model(patches, geoms)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            
        # Val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                patches, geoms, labels = batch
                patches = patches.to(device)
                geoms = geoms.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                logits = model(patches, geoms)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_acc = correct / total if total > 0 else 0
        print(f"Epoch {ep+1} Loss: {loss_sum/len(train_loader):.4f} Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config['train']['save_dir'], "best_edge_model.pth"))

if __name__ == "__main__":
    train()

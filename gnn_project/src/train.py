import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import argparse
import glob
from tqdm import tqdm
from models import NetlistGNN
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

class InMemoryGraphDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(root_dir, '*.pt')))
        
    def len(self):
        return len(self.files)

    def get(self, idx):
        # Using weights_only=False because PyG Data objects contain custom classes
        return torch.load(self.files[idx], weights_only=False)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    full_dataset = InMemoryGraphDataset(args.data_root)
    
    if len(full_dataset) == 0:
        print(f"❌ Error: No .pt files found in {args.data_root}")
        print("Please run 'python gnn_project/src/prepare_dataset.py' first.")
        return

    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    if train_size == 0:
        # Not enough data for split, use all for both or handle
        train_dataset = full_dataset
        val_dataset = full_dataset
        print("⚠️ Warning: Very small dataset, using same data for train/val.")
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = NetlistGNN(num_classes=args.num_classes, model_type=args.model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss (Weighted BCE for imbalance)
    # pos_weight calculated roughly or passed as arg. Assuming rare connections:
    pos_weight = torch.tensor([5.0]).to(device) 
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_f1 = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            data = data.to(device)
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: Loss {total_loss:.4f} | F1: {val_metrics['f1']:.4f} | Prec: {val_metrics['prec']:.4f} | Rec: {val_metrics['rec']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print("Saved best model.")

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            
    f1 = f1_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, zero_division=0)
    r = recall_score(all_labels, all_preds, zero_division=0)
    
    return {'f1': f1, 'prec': p, 'rec': r}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to processed .pt files directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model_type', type=str, default='SAGE')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)

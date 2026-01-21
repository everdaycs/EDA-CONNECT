import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import glob
from tqdm import tqdm
from model import EdgePredGNN
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class CircuitDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.root_dir = root
        super(CircuitDataset, self).__init__(root, transform, pre_transform)
        self.file_list = sorted(glob.glob(os.path.join(root, '*.pt')))

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        return torch.load(self.file_list[idx], weights_only=False)

def train():
    # Config
    DATA_PATH = '/home/kaga/Desktop/EDA-Connect/gnn_project/processed_data/raw_graphs'
    BATCH_SIZE = 8 # Graphs per batch (variable edges)
    LR = 0.001
    EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data Setup
    dataset = CircuitDataset(DATA_PATH)
    
    # Split
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    # Node feat dim = 6 (x,y,w,h,is_w,is_c)
    model = EdgePredGNN(in_channels=6, hidden_channels=32).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Weighted Loss for imbalance?
    # Ideally should calculate pos_weight dynamically, but let's start with standard BCE
    # or a fixed weight if we know ratio.
    pos_weight = torch.tensor([5.0]).to(DEVICE) # Assume negs are 5x more frequent
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    
    best_f1 = 0.0
    
    print("Starting GNN Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            
            # out shape [num_edges, 1], y shape [num_edges]
            loss = criterion(out.view(-1), batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(out).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(batch.y.cpu().numpy())
            
        avg_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                
                probs = torch.sigmoid(out).view(-1).cpu()
                preds = (probs > 0.5).int()
                
                val_probs.extend(probs.numpy())
                val_preds.extend(preds.numpy())
                val_labels.extend(batch.y.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, zero_division=0)
        val_rec = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except:
            val_auc = 0.0
            
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | Train F1 {train_f1:.4f} | Val Acc {val_acc:.4f} F1 {val_f1:.4f} AUC {val_auc:.3f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), '/home/kaga/Desktop/EDA-Connect/gnn_project/checkpoints/best_gnn.pth')
            print(">>> Saved Best Model")

if __name__ == "__main__":
    train()

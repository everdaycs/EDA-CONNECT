import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ViTSkeleton
from datasets import SkeletonDataset
from losses import CompositeLoss

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    dataset = SkeletonDataset(args.img_dir, args.skel_dir, img_size=args.img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    
    model = ViTSkeleton(model_name='deit_tiny_patch16_224', img_size=args.img_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = CompositeLoss()
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for img, mask in pbar:
            img, mask = img.to(device), mask.to(device)
            
            optimizer.zero_grad()
            out = model(img)
            
            loss = criterion(out, mask)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pt"))
            print("Saved Best Model.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--skel_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='vit_skel_project/checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=224) # Increased default? Deit likes 224 usually
    args = parser.parse_args()
    
    train(args)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class NetlistGNN(torch.nn.Module):
    def __init__(self, num_classes, hidden_dim=64, out_dim=64, model_type='SAGE'):
        super(NetlistGNN, self).__init__()
        
        # 1. Node Encoder
        # Input feat: [ClassID, cx, cy, w, h] -> We need to embed ClassID
        self.class_emb = nn.Embedding(num_classes, hidden_dim)
        self.geom_encoder = nn.Linear(4, hidden_dim) 
        
        input_dim = hidden_dim * 2 # Concat class emb + geom feat
        
        self.convs = nn.ModuleList()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        if model_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, out_dim))
        elif model_type == 'SAGE':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, out_dim))
        elif model_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
            self.convs.append(GATConv(hidden_dim, out_dim, heads=4, concat=False))
            
        # 2. Edge Classifier (Link Prediction Head)
        # Input: concat(h_u, h_v) or similar
        self.edge_mlp = nn.Sequential(
            nn.Linear(out_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index):
        # x shape: [N, 5] (cls, cx, cy, w, h)
        
        # Split features
        cls_idx = x[:, 0].long()
        geom_feat = x[:, 1:]
        
        # Encode
        h_cls = self.class_emb(cls_idx)
        h_geom = self.geom_encoder(geom_feat)
        
        # Fuse
        h = torch.cat([h_cls, h_geom], dim=1) # [N, hidden*2]
        
        # GNN Layers
        for conv in self.convs[:-1]:
            h = conv(h, edge_index)
            h = self.act(h)
            h = self.dropout(h)
            
        h = self.convs[-1](h, edge_index)
        # h is now node embedding [N, out_dim]
        
        # Edge Prediction
        # We need embeddings for source and target nodes of edge_index
        row, col = edge_index
        
        h_src = h[row]
        h_dst = h[col]
        
        # Combine strategies: concat, hadamard, L1...
        # Using concat
        edge_feat = torch.cat([h_src, h_dst], dim=1)
        
        out = self.edge_mlp(edge_feat)
        return out.squeeze(-1) # Logits


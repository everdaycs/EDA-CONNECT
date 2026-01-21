import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class EdgePredGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(EdgePredGNN, self).__init__()
        
        # GNN Backbone
        # Use GAT for handling different neighbor importance
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True)
        self.conv3 = GATConv(hidden_channels * 4, hidden_channels * 2, heads=1, concat=False) # Output dim: hidden_channels*2
        
        self.node_embedding_dim = hidden_channels * 2
        
        # Edge Classifier MLP
        # Input: Node_i (dim) + Node_j (dim) + EdgeAttr (2: dist, iou)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * self.node_embedding_dim + 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: Node features
        # edge_index: Graph connectivity (candidates)
        # edge_attr: Edge features (dist, iou)
        
        # 1. Message Passing to update node features
        # GATConv typically doesn't use edge attributes in the aggregation unless specified.
        # For simplicity, we stick to node feature propagation based on connectivity.
        
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        x = self.conv3(x, edge_index)
        # x is now the learned node embedding
        
        # 2. Edge Classification
        # Extract source and target node embeddings for each edge
        row, col = edge_index
        x_i = x[row] # Source node embeddings
        x_j = x[col] # Target node embeddings
        
        # Cat (Node_i, Node_j, EdgeFeature)
        edge_feat_cat = torch.cat([x_i, x_j, edge_attr], dim=1)
        
        out = self.edge_mlp(edge_feat_cat)
        
        return out


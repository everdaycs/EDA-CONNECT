import torch
import torch.nn as nn
import torchvision.models as models

class EdgeCNN(nn.Module):
    def __init__(self, geom_dim=8, hidden_dim=64):
        super(EdgeCNN, self).__init__()
        
        # Backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove FC
        self.encoder = nn.Sequential(*list(resnet.children())[:-1]) 
        self.embed_dim = 512 # ResNet18 output dim
        
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(self.embed_dim + geom_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Logit output
        )

    def forward(self, patch, geom):
        # Patch: [B, 3, H, W]
        # Geom: [B, 8]
        
        features = self.encoder(patch) # [B, 512, 1, 1]
        features = features.view(features.size(0), -1)
        
        combined = torch.cat([features, geom], dim=1)
        
        logit = self.fusion(combined)
        return logit

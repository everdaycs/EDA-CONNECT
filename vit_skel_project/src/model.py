import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ViTSkeleton(nn.Module):
    def __init__(self, model_name='deit_tiny_patch16_224', img_size=224, num_classes=1):
        super(ViTSkeleton, self).__init__()
        # Load backbone
        self.backbone = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
        
        # Get embedding info (robust way)
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
            features = self.backbone.forward_features(dummy)
            # Timm ViT features: [B, N, C] or [B, C, H, W] depending on model
            # For DeiT/ViT it's [B, N+dist, C] usually
            self.embed_dim = features.shape[-1]
            
        # Simple Decoder
        # 1. Project back to spatial
        self.norm = nn.BatchNorm2d(self.embed_dim)
        
        # 2. Upsampling blocks
        # Assuming patch size 16, we need to upsample 16x
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32), # Now 16x larger
            
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Features: [B, N, D] (CLS token + patches)
        # Handle different timm versions/models
        feat = self.backbone.forward_features(x)
        
        # If output is sequence [B, L, D] (ViT/DeiT)
        if len(feat.shape) == 3:
            # removing CLS (and Distillation token if exists)
            # This is heuristic, for standard ViT-patch16
            n_patches = feat.shape[1]
            # Heuristic to determine grid size
            grid_h = H // 16
            grid_w = W // 16
            
            # Check if distillation token exists (DeiT has 2 extra tokens often)
            num_extra = n_patches - (grid_h * grid_w)
            
            patch_feat = feat[:, num_extra:, :] # [B, H*W, D]
            patch_feat = patch_feat.permute(0, 2, 1).reshape(B, self.embed_dim, grid_h, grid_w)
        else:
            # If ConvNeXt or hierarchical, it's already [B, C, H, W]
            patch_feat = feat
            
        x_dec = self.norm(patch_feat)
        out = self.decoder(x_dec)
        
        # Interpolate if exact match needed (e.g. if input wasn't multiple of 16)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            
        return out

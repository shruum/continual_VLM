import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from timm.models.vision_transformer import VisionTransformer

# Initialize CLIP Vision Model (ViT-B/32) from Scratch
class CLIPVisionClassifier(nn.Module):
    def __init__(self, img_size=224, patch_size=32, embed_dim=768, num_classes=10):
        super(CLIPVisionClassifier, self).__init__()
        self.backbone = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=embed_dim,
            embed_dim=embed_dim,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=nn.LayerNorm
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
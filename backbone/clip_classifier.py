import torch
import torch.nn as nn

class ClipClassifier(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes):
        super(ClipClassifier, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():  # No gradient for backbone
            features = self.backbone(x)
        return self.fc(features)
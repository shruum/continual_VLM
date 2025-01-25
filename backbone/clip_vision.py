import clip
import torch


class CLIPFeatureExtractor(torch.nn.Module):
    def __init__(self, clip_model):
        super(CLIPFeatureExtractor, self).__init__()
        self.clip_model = clip_model.visual  # Visual part of CLIP (ViT encoder)
        self.full_model = clip_model  # Full model with text and image encoding

    def forward(self, x, returnt='out'):
        # Preprocess the input through the visual layers (like patch embedding, etc.)
        x = self.clip_model.conv1(x)  # Apply patch embedding
        x = x.reshape(x.shape[0], x.shape[1], -1)  # Flatten patches
        x = x.permute(0, 2, 1)  # Reshape for transformer
        x = torch.cat([self.clip_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                 dtype=x.dtype, device=x.device), x],
                      dim=1)  # Add class token
        x = x + self.clip_model.positional_embedding.to(x.dtype)  # Add positional embedding
        x = self.clip_model.ln_pre(x)  # Pre-layer normalization

        # Pass through the transformer layers
        x = x.permute(1, 0, 2)  # Reshape for transformer
        x = self.clip_model.transformer(x)  # Transformer layers
        x = x.permute(1, 0, 2)  # Reshape back

        # Extract features from the class token (index 0 corresponds to the class token)
        features = self.clip_model.ln_post(x[:, 0, :])  # Layer norm on class token
        if self.clip_model.proj is not None:
            features = features @ self.clip_model.proj  # Final projection if it exists

        # Pass features through the classification layer to get logits
        outputs = features @ self.full_model.token_embedding.weight.T  # CLIP classification logits

        if returnt == 'all':
            return outputs, features  # Return both logits (outputs) and features
        elif returnt == 'out':
            return outputs
import torch
import torch.nn as nn

from utils.aux_utils import get_clip_embeddings



def loss_vlm(model, labels, class_names, features):

    all_text_features = get_clip_embeddings(model.text_encoder, labels, model.device, class_names)
    all_text_features = all_text_features.to(model.device)
    loss = 0
    if model.args.loss_mode == 'l2':
        # all_text_features = torch.stack([text_emb for text_emb in all_text_emb])
        loss_aux12 = model.aux.loss(features, all_text_features)
        loss = (loss_aux12 * model.args.loss_wt[0])

    elif model.args.loss_mode == 'nce':
        dim = 1024
        # projection MLP
        proj1 = nn.Sequential(
            nn.Linear(features.size(1), dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, affine=False),
        )
        proj1 = proj1.to(model.device)
        # predictor MLP
        proj2 = nn.Sequential(
            nn.Linear(dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, dim),
        )
        proj2 = proj2.to(model.device)

        fx = torch.flatten(features, start_dim=1)
        zx = proj1(fx)
        px = proj2(zx)

        if features.size(1) != all_text_features.size(1):
            proj1 = nn.Sequential(
                nn.Linear(all_text_features.size(1), dim, bias=False),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim, affine=False),
            ).to(model.device)
        fy = torch.flatten(all_text_features, start_dim=1)
        zy = proj1(fy)
        # py = self.proj2(zy)
        loss_aux = -(model.nce_criterion(px, zy.detach()).mean())
        loss = (loss_aux * model.args.loss_wt[0])

    return loss




import torch
import torch.nn.functional as F
from torch import nn

from models.text.text_enc import get_text_embeddings
# from utils.cka import cka_similarity

TEX_DIM = {
    "sent_transf": 384,
    "bert": 768,
    "clip": 512
}

class lossVLM():

    def __init__(self, model):


        self.model = model
        self.device = model.device
        self.rev_proj = self.model.args.rev_proj

        self.img_dim = 2048
        self.img_hdim = 4096
        self.text_hdim = 1024
        self.text_dim = TEX_DIM.get(self.model.args.text_model)

        self.proj_i = nn.Sequential(
            nn.Linear(self.img_dim, self.img_hdim, bias=False),
            nn.BatchNorm1d(self.img_hdim),
            nn.ReLU(inplace=True),
            nn.Linear(self.img_hdim, self.text_dim),
            # nn.BatchNorm1d(self.text_dim, affine=False),
        ).to(self.device)
        # predictor MLP
        self.pred_i = nn.Sequential(
            nn.Linear(self.text_dim, self.text_hdim, bias=False),
            nn.BatchNorm1d(self.text_hdim),
            nn.ReLU(inplace=True),
            nn.Linear(self.text_hdim, self.text_dim),
        ).to(self.device)


        self.proj_t = nn.Sequential(
            nn.Linear(self.text_dim, self.text_hdim, bias=False),
            nn.BatchNorm1d(self.text_hdim),
            nn.ReLU(inplace=True),
            nn.Linear(self.text_hdim, self.img_dim),
        ).to(self.device)
        # predictor MLP
        self.pred_t = nn.Sequential(
            nn.Linear(self.img_dim, self.img_hdim, bias=False),
            nn.BatchNorm1d(self.img_hdim),
            nn.ReLU(inplace=True),
            nn.Linear(self.img_hdim, self.img_dim),
        ).to(self.device)

    def loss_vlm(self, labels, dataset, features, dataloader=None):

        all_text_features = get_text_embeddings(self.model, labels, dataset, dataloader)
        all_text_features = all_text_features.to(self.device)

        loss = 0
        if self.model.args.loss_mode == 'l2':
            # all_text_features = torch.stack([text_emb for text_emb in all_text_emb])
            if self.rev_proj:
                features = self.proj_i(features)
            else:
                all_text_features = self.proj_t(all_text_features)
            loss_aux12 = self.l2_loss(features, all_text_features)
            loss = (loss_aux12 * self.model.args.loss_wt[0])

        elif self.model.args.loss_mode == 'kl':
            # all_text_features = torch.stack([text_emb for text_emb in all_text_emb])
            if self.rev_proj:
                features = self.proj_i(features)
            else:
                all_text_features = self.proj_t(all_text_features)
            loss_aux12 = self.kl_loss(features, all_text_features)
            loss = (loss_aux12 * self.model.args.loss_wt[0])

        elif self.model.args.loss_mode == 'nce':
            # projection MLP
            fx = torch.flatten(features, start_dim=1)
            fy = torch.flatten(all_text_features, start_dim=1)
            # zx = self.proj_i(fx)
            # if features.size(1) != all_text_features.size(1):
            if self.rev_proj:
                px = self.proj_i(fx)
                zx = self.pred_i(px)
                loss_aux = - (self.nce_loss(zx, fy))
            else:
                py = self.proj_t(fy)
                zy = self.pred_t(py)
                loss_aux = - (self.nce_loss(fx, zy))
            loss = (loss_aux * self.model.args.loss_wt[0])

        elif self.model.args.loss_mode == 'sim':
            # all_text_features = torch.stack([text_emb for text_emb in all_text_emb])
            loss_aux12 = self.similarity_preserving_loss(features, all_text_features)
            loss = (loss_aux12 * self.model.args.loss_wt[0])

        # similarity_score_lin = cka_similarity(features, all_text_features, sim_type='linear')
        # similarity_score_ker = cka_similarity(features, all_text_features, sim_type='kernel')
        # print(f"CKA Similarity Linear: {similarity_score_lin}")
        # print(f"CKA Similarity Kernel: {similarity_score_ker}")

        return loss

    def kl_loss(self, out1, out2, T=1):
        p = F.log_softmax(out1 / T, dim=1)
        q = F.softmax(out2 / T, dim=1)
        l_kl = F.kl_div(p, q, size_average=False) * (T**2) / out1.shape[0]
        return l_kl

    def l2_loss(self, out1, out2):
        criterion_MSE = nn.MSELoss(reduction='mean')
        return criterion_MSE(out1, out2)

    def nce_loss(self, out1, out2):
        nce_criterion = (nn.CosineSimilarity(dim=1).cuda(self.device))
        return nce_criterion(out1, out2.detach()).mean()
    def similarity_preserving_loss(self, A_t, A_s):
        """Given the activations for a batch of input from the teacher and student
        network, calculate the similarity preserving knowledge distillation loss from the
        paper Similarity-Preserving Knowledge Distillation (https://arxiv.org/abs/1907.09682)
        equation 4
        Note: A_t and A_s must have the same batch size
        Parameters:
            A_t (4D tensor): activation maps from the teacher network of shape b x c1 x h1 x w1
            A_s (4D tensor): activation maps from the student network of shape b x c2 x h2 x w2
        Returns:
            l_sp (1D tensor): similarity preserving loss value
    """
        # reshape the activations
        # b1, c1, h1, w1 = A_t.shape
        # b2, c2, h2, w2 = A_s.shape
        # assert b1 == b2, 'Dim0 (batch size) of the activation maps must be compatible'
        Q_t = A_t #.reshape([b1, c1 * h1 * w1])
        Q_s = A_s #.reshape([b2, c2 * h2 * w2])

        # evaluate normalized similarity matrices (eq 3)
        G_t = torch.mm(Q_t, Q_t.t())
        # G_t = G_t / G_t.norm(p=2)
        G_t = torch.nn.functional.normalize(G_t)

        G_s = torch.mm(Q_s, Q_s.t())
        # G_s = G_s / G_s.norm(p=2)
        G_s = torch.nn.functional.normalize(G_s)

        # calculate the similarity preserving loss (eq 4)
        l_sp = (G_t - G_s).pow(2).mean()

        return l_sp


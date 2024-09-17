# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from utils.args import *
from utils.vision_lang import lossVLM
from tqdm import tqdm
import torch.nn.functional as F
from models.text.text_enc import TextEncoder


def cross_entropy(y, labels):
    l_ce = F.cross_entropy(y, labels)
    return l_ce

class Normal:
    NAME = 'normal'

    def __init__(self, args, backbone, dataset, device):
        # super(Normal, self).__init__(args, backbone, dataset)
        self.backbone = backbone
        self.args = args
        self.dataset = dataset
        self.device = device
        if self.args.mode == 'vlm':
            self.vlm_loss = lossVLM(self)
            self.text_model = self.args.text_model
            self.text_encoder = TextEncoder(self.args.text_model, device=self.device, pretrain=True)

    def train_normal(self, train_loader, optimizer, epoch):

        self.backbone.train()
        train_loss = 0
        correct = 0
        total = 0
        num_batches = len(train_loader)

        for batch_idx, (data, target) in tqdm(enumerate(train_loader), desc='batch training', total=num_batches):

            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()

            out, features = self.backbone(data, returnt='all')
            iteration = (epoch * num_batches) + batch_idx

            if self.args.mode == 'normal':
                loss = cross_entropy(out, target)
            elif self.args.mode == 'vlm':
                loss = cross_entropy(out, target)
                loss += self.vlm_loss.loss_vlm(target, self.dataset, features)
            # perform back propagation
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()
            _, predicted = torch.max(out.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().float().sum()
            b_idx = batch_idx
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

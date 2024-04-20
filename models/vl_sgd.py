# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch.nn as nn
from utils.aux_utils import AuxiliaryNet
from utils.aux_utils import get_clip_embeddings

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Vision and language Continual learning without Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_auxiliary_args(parser)

    return parser


class VLSgd(ContinualModel):
    NAME = 'vl_sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual', 'multi-modal']

    def __init__(self, backbone, loss, args, transform):
        super(VLSgd, self).__init__(backbone, loss, args, transform)
        self.aux = AuxiliaryNet(self.args, self.device)
        self.addit_models = ['net']

        self.task = 0
        self.iteration = 0

    def observe(self, inputs, labels, not_aug_inputs, class_names=None):

        loss_dict = {}
        self.opt.zero_grad()

        outputs, features = self.net(inputs, returnt='all')
        # labels_hot = nn.functional.one_hot(labels, num_classes=10).float()
        loss_ce1 = self.loss(outputs, labels)

        all_text_features = get_clip_embeddings(self.text_encoder, labels, self.device, class_names)
        all_text_features = all_text_features.to(self.device)
        # all_text_features = torch.stack([text_emb for text_emb in all_text_emb])
        loss_aux12 = self.aux.loss(features, all_text_features)

        loss = loss_ce1 + (loss_aux12 * self.args.loss_wt[0])

        self.aux.collate_loss(loss_dict, loss_ce=loss_ce1, loss_aux=loss_aux12, m1=True)
        if hasattr(self, 'writer'):
            for loss_name, loss_item in loss_dict.items():
                self.writer.add_scalar('Task {}/{}'.format(self.task, loss_name), loss_item,
                                          global_step=self.iteration)

        loss.backward()
        self.opt.step()


        return loss.item()

    def end_task(self, dataset):
        print('Saving Model')
        self.task += 1
        self.save_models(dataset)






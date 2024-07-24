# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from models.utils.continual_model import ContinualModel
from utils.args import *
from utils.buffer import Buffer
import torch.nn as nn
from utils.aux_utils import AuxiliaryNet
from models.text.text_enc import get_text_embeddings
from torch.nn import functional as F

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Vision and language Continual learning via Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_auxiliary_args(parser)

    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')

    return parser


class VLDer(ContinualModel):
    NAME = 'vl_der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(VLDer, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.aux = AuxiliaryNet(self.args, self.device)
        self.addit_models = ['net']

        self.task = 0
        self.iteration = 0

    def observe(self, inputs, labels, not_aug_inputs, class_names=None):

        loss_dict = {}
        loss_log_12 = 0

        self.opt.zero_grad()
        outputs, features = self.net(inputs, returnt='all')
        loss_ce1 = loss = self.loss(outputs, labels)

        all_text_features = get_text_embeddings(self.text_encoder, labels, self.device, class_names)
        all_text_features = all_text_features.to(self.device)
        loss_aux12 = self.aux.loss(features, all_text_features)
        loss += (loss_aux12 * self.args.loss_wt[0])

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss_log_12 = F.mse_loss(buf_outputs, buf_logits)
            loss += (self.args.alpha * loss_log_12)

        self.aux.collate_loss(loss_dict, loss_ce=loss_ce1, loss_aux=loss_aux12, loss_logit_mem=loss_log_12, m1=True)

        if hasattr(self, 'writer'):
            for loss_name, loss_item in loss_dict.items():
                self.writer.add_scalar('Task {}/{}'.format(self.task, loss_name), loss_item,
                                          global_step=self.iteration)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return loss.item()

    def end_task(self, dataset):
        print('Saving Model')
        self.task += 1
        self.save_models(dataset)







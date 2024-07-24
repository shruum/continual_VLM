# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import json
from models.utils.continual_model import ContinualModel
from utils.args import *
from utils.buffer import Buffer
import torch.nn as nn
from utils.aux_utils import AuxiliaryNet
from models.text.text_enc import get_text_embeddings

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
}
CLASS_ID = {0:"car (automobile)", 1:"airplane", 2:"bird", 3:"cat", 4:"deer", 5:"dog",
            6:"frog", 7:"horse", 8:"passenger ship", 9:"truck"}


def get_parser() -> ArgumentParser:
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_auxiliary_args(parser)

    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')

    return parser


class VLDerpp(ContinualModel):
    NAME = 'vl_derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.aux = AuxiliaryNet(self.args, self.device)
        self.addit_models = ['net']

        self.task = 0
        self.iteration = 0

    def observe(self, inputs, labels, not_aug_inputs):

        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss_ce1 = loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss_log_12 = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss_buf_ce1 = self.args.beta * self.loss(buf_outputs, buf_labels)

            loss += (self.args.alpha_mm[0] * loss_log_12) + (self.args.beta_mm[0] * loss_buf_ce1)


        all_text_features = get_text_embeddings(self.text_encoder, labels, self.device)
        all_text_features = all_text_features.to(self.device)
        # all_text_features = torch.stack([text_emb for text_emb in all_text_emb])
        loss_aux12 = self.aux.loss(features, all_text_features)
        loss += (loss_aux12 * self.args.loss_wt[0])

        self.aux.collate_loss(loss_dict, loss_ce=loss_ce1, loss_buf_ce=loss_buf_ce1, loss_aux=loss_aux12, loss_logit_mem=loss_log_12, m1=True)

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







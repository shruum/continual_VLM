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
from utils.vision_lang import lossVLM

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Vision and language Continual learning via Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_auxiliary_args(parser)

    return parser


class VLER(ContinualModel):
    NAME = 'vl_er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual', 'multi-modal']

    def __init__(self, backbone, loss, args, transform):
        super(VLER, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.aux = AuxiliaryNet(self.args, self.device)
        self.addit_models = ['net']

        self.task = 0
        self.iteration = 0
        self.kd_loss = lossVLM(self)

    def observe(self, inputs, labels, not_aug_inputs, dataset=None):

        real_batch_size = inputs.shape[0]
        loss = 0
        loss_dict = {}
        self.opt.zero_grad()

        outputs, features = self.net(inputs, returnt='all')
        loss_aux = self.kd_loss.loss_vlm(labels, dataset, features)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss_ce1 = self.loss(outputs, labels)

        loss = loss_ce1 + loss_aux

        self.aux.collate_loss(loss_dict, loss_ce=loss_ce1, loss_aux=loss_aux, m1=True)
        if hasattr(self, 'writer'):
            for loss_name, loss_item in loss_dict.items():
                self.writer.add_scalar('Task {}/{}'.format(self.task, loss_name), loss_item,
                                          global_step=self.iteration)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset):
        print('Saving Model')
        self.task += 1
        self.save_models(dataset)






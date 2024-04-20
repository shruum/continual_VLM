# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Type

import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.MNISTMLP import MNISTMLP
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import os
from datasets.transforms.permutation import Permutation
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.validation import get_train_val
# from utils.conf import base_path_dataset as base_path
from argparse import Namespace
import numpy as np


def store_mnist_loaders(transform, setting):
    train_dataset = MyMNIST(os.path.join(setting.args.dataset_dir, 'MNIST'),
                            train=True, download=True, transform=transform)
    if setting.args.validation:
        train_dataset, test_dataset = get_train_val(train_dataset,
                                                    transform, setting.NAME)
    else:
        test_dataset = MNIST(os.path.join(setting.args.dataset_dir, 'MNIST'),
                             train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    return train_loader, test_loader


class MyMNIST(MNIST):
    """
    Overrides the MNIST dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        super(MyMNIST, self).__init__(root, train, transform,
                                      target_transform, download)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, img


class PermutedMNIST(ContinualDataset):

    NAME = 'perm-mnist'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 20

    def __init__(self, args: Namespace) -> None:
        super(PermutedMNIST, self).__init__(args)
        np.random.seed(args.mnist_seed)
        PermutedMNIST.N_TASKS = args.n_tasks_mnist
        self.permutations = [Permutation() for _ in range(PermutedMNIST.N_TASKS)]
        self.task_id = 0

    def get_data_loaders(self):
        transform = transforms.Compose((transforms.ToTensor(), self.permutations[self.task_id]))
        train, test = store_mnist_loaders(transform, self)
        self.task_id += 1
        return train, test

    @staticmethod
    def get_backbone():
        return MNISTMLP(28 * 28, PermutedMNIST.N_CLASSES_PER_TASK)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_scheduler(model, args):
        return None

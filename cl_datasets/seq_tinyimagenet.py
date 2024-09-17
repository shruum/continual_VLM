# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset
from backbone.ResNet18 import resnet18, resnet50
from backbone.ResNet_mam import resnet18mam, resnet50mam
import torch.nn.functional as F
# from utils.conf import base_path_dataset as base_path
from PIL import Image
import os
from cl_datasets.utils.validation import get_train_val
from cl_datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from cl_datasets.transforms.denormalization import DeNormalize
from cl_datasets.utils.tinyimg_split import divide_into_tasks

class ImageFolderTest(datasets.ImageFolder):
    def __init__(self, root, files_list, transform=None, target_transform=None,
                 loader=datasets.folder.default_loader, classes=None, class_to_idx=None, imgs=None, not_aug_transform=None):
        """
        :param root: root path of the dataset
        :param files_list: list of filenames to include in this dataset
        :param classes: classes to include, based on subdirs of root if None
        :param class_to_idx: overwrite class to idx mapping
        :param imgs: list of image paths (under root)
        """
        if classes is None:
            assert class_to_idx is None
            classes, class_to_idx = self.find_classes(root)
        elif class_to_idx is None:
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        imgs = self.make_dataset(root, class_to_idx, files_list) if imgs is None else imgs
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in subfolders of: {}\nSupported image extensions are: {}".
                               format(root, ",".join(datasets.folder.IMG_EXTENSIONS)))
        self.root = root
        self.samples = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.not_aug_transform = not_aug_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
          return img, target, original_img, self.logits[index]

        return img, target,

class ImageFolderTrain(datasets.ImageFolder):
    def __init__(self, root, files_list, transform=None, target_transform=None,
                 loader=datasets.folder.default_loader, classes=None, class_to_idx=None, imgs=None, not_aug_transform=None):
        """
        :param root: root path of the dataset
        :param files_list: list of filenames to include in this dataset
        :param classes: classes to include, based on subdirs of root if None
        :param class_to_idx: overwrite class to idx mapping
        :param imgs: list of image paths (under root)
        """
        if classes is None:
            assert class_to_idx is None
            classes, class_to_idx = self.find_classes(root)
        elif class_to_idx is None:
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        imgs = self.make_dataset(root, class_to_idx, files_list) if imgs is None else imgs
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in subfolders of: {}\nSupported image extensions are: {}".
                               format(root, ",".join(datasets.folder.IMG_EXTENSIONS)))
        self.root = root
        self.samples = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.not_aug_transform = not_aug_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
          return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img
class SequentialTinyImagenet(ContinualDataset):

    NAME = 'seq-tinyimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(64, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4802, 0.4480, 0.3975),
                                  (0.2770, 0.2691, 0.2821))])
    # Read the contents of the words and wnids files
    CLASS_ID = {}
    with open('cl_datasets/metadata/wnids.txt', 'r') as file:
        for idx, line in enumerate(file):
            CLASS_ID[idx] = line.strip()

    def __init__(self,args):

        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

        self.train_dataset = {}
        self.test_dataset = {}
        transform = self.TRANSFORM
        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])
        not_aug_transform = transforms.Compose([transforms.ToTensor()])

        task_count = self.N_TASKS
        self.img_paths = divide_into_tasks(args.dataset_dir, task_count)
        for task in range(1, task_count + 1):
            print("\nTASK ", task)
            self.train_dataset[task] = ImageFolderTrain(
                os.path.join(args.dataset_dir, 'train'), None, transform=transform,
                classes=self.img_paths[task]['classes'], class_to_idx=self.img_paths[task]['class_to_idx'],
                imgs=self.img_paths[task]['train'], not_aug_transform=not_aug_transform
            )

            self.test_dataset[task] = ImageFolderTest(
                os.path.join(args.dataset_dir, 'test'), None, transform=test_transform,
                classes=self.img_paths[task]['classes'], class_to_idx=self.img_paths[task]['class_to_idx'],
                imgs=self.img_paths[task]['val']
            )

    def get_data_loaders(self):

        task = self.i + 1
        transform = self.TRANSFORM
        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train = torch.utils.data.DataLoader(self.train_dataset[task], batch_size=self.args.batch_size,
                                                       shuffle=True, num_workers=8)
        test = torch.utils.data.DataLoader(self.test_dataset[task], batch_size=self.args.batch_size,
                                                   shuffle=False, num_workers=8)

        self.test_loaders.append(test)
        self.train_loader = train
        self.i += 1 #self.N_CLASSES_PER_TASK
        return train, test

    def get_backbone(self):
        if self.args.arch == 'resnet18':
            return resnet18(SequentialTinyImagenet.N_CLASSES_PER_TASK
                        * SequentialTinyImagenet.N_TASKS)
        elif self.args.arch == 'resnet50':
            return resnet50(SequentialTinyImagenet.N_CLASSES_PER_TASK
                        * SequentialTinyImagenet.N_TASKS)
        elif self.args.arch == 'resnet18mam':
            return resnet18mam(SequentialTinyImagenet.N_CLASSES_PER_TASK
                        * SequentialTinyImagenet.N_TASKS)
        elif self.args.arch == 'resnet50mam':
            return resnet50mam(SequentialTinyImagenet.N_CLASSES_PER_TASK
                        * SequentialTinyImagenet.N_TASKS)
        else:
            raise (RuntimeError("architecture type not found"))

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialTinyImagenet.get_batch_size()

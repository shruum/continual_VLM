import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from backbone.ResNet18 import resnet50, resnet18
from backbone.ResNet_mam import resnet18mam, resnet50mam
import torch.nn.functional as F
# from utils.conf import base_path_img
from cl_datasets.utils.continual_dataset import ContinualDataset, store_domain_loaders
from cl_datasets.transforms.denormalization import DeNormalize
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(flist, sep):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split(sep)
            imlist.append((impath, int(imlabel)))

    return imlist

class ImageFilelist(Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None, not_aug_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader, sep=' '):
        self.root = root
        self.imlist = flist_reader(flist, sep)
        self.targets = np.array([datapoint[1] for datapoint in self.imlist])
        self.data = np.array([datapoint[0] for datapoint in self.imlist])
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.not_aug_transform = not_aug_transform

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        original_img = img.copy()

        if self.not_aug_transform:
            not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
          return img, target, not_aug_img, self.logits[index]

        if self.not_aug_transform:
            return img, target, not_aug_img
        else:
            return img, target


    def __len__(self):
        return len(self.imlist)

class DN4IL(ContinualDataset):

    NAME = 'dn4il'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 100
    N_TASKS = 6
    IMG_SIZE = 64
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]
    DOMAIN_LST = [ 'real', 'clipart', 'infograph', 'painting', 'sketch', 'quickdraw']

    normalize = transforms.Normalize((0.485, 0.456, 0.406),
                                      (0.229, 0.224, 0.225))
    TRANSFORM = [transforms.Resize((IMG_SIZE, IMG_SIZE)),
                 transforms.RandomCrop(IMG_SIZE, padding=4),  # transforms.RandomResizedCrop(IMG_SIZE),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 normalize]
    TRANSFORM_TEST = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                      normalize]
    NOT_AUG_TRANSFORM = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]

    def __init__(self, args):
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

        self.transform = transforms.Compose(self.TRANSFORM)
        self.test_transform = transforms.Compose(self.TRANSFORM_TEST)
        self.not_aug_transform = transforms.Compose(self.NOT_AUG_TRANSFORM)

        self.data_path = os.path.join(args.dataset_dir, 'DomainNet')
        self.annot_path = os.path.join(args.dataset_dir)

    def get_data_loaders(self):

        train_dataset = ImageFilelist(
            root=self.data_path,
            flist=os.path.join(self.annot_path, self.DOMAIN_LST[self.i] + "_train.txt"),
            transform=self.transform,
            not_aug_transform=self.not_aug_transform,
            )

        test_dataset = ImageFilelist(
            root=self.data_path,
            flist=os.path.join(self.annot_path, self.DOMAIN_LST[self.i] + "_test.txt"),
            transform=self.test_transform,
            )

        train, test = store_domain_loaders(train_dataset, test_dataset, self)

        return train, test

    def not_aug_dataloader(self, batch_size):
        pass
        # return DataLoader(self.train_loader.dataset,
        #                   batch_size=batch_size, shuffle=True)

    def get_backbone(self):
        if self.args.arch == 'resnet18':
            return resnet18(DN4IL.N_CLASSES_PER_TASK)
        elif self.args.arch == 'resnet50':
            return resnet50(DN4IL.N_CLASSES_PER_TASK)
        elif self.args.arch == 'resnet18mam':
            return resnet18mam(DN4IL.N_CLASSES_PER_TASK)
        elif self.args.arch == 'resnet50mam':
            return resnet50mam(DN4IL.N_CLASSES_PER_TASK)
        else:
            raise (RuntimeError("architecture type not found"))

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Compose(self.TRANSFORM)])
        return transform


    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
        return transform
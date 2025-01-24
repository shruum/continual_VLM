import os
import torch
import pandas as pd
import json
import torchvision
from torchvision import datasets, transforms
from norm_datasets.utils import celeb_indicies, cif_tint, MappedImageFolder, waterbird_shortcut
from norm_datasets.cifar_imbalance import CIFAR10ImbalancedNoisy
from torch.utils.data import DataLoader, ConcatDataset

class CIFAR10:
    """
    CIFAR-10 dataset
    """
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    CLASS_ID = {0: "car (automobile)", 1: "airplane", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                6: "frog", 7: "horse", 8: "cargo ship", 9: "truck"}
    NUM_CLASSES = 10
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    SIZE = 32

    def __init__(self, data_path, **kwargs):
        self.data_path = data_path
        self.arch = kwargs.get('arch', None)

        if 'clip' in self.arch:
            self.transform_train = transforms.Compose(
                [transforms.Resize((224, 224)),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=CIFAR10.MEAN, std=CIFAR10.STD)
                 ])
            self.transform_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10.MEAN,std=CIFAR10.STD)
            ])
        else:
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(CIFAR10.SIZE, padding=4),  # Random crop with padding for augmentation
                transforms.RandomHorizontalFlip(),  # Random horizontal flip
                transforms.ToTensor(),  # Convert image to tensor
                transforms.Normalize(mean=CIFAR10.MEAN,std=CIFAR10.STD)
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10.MEAN,std=CIFAR10.STD)
            ])

    def get_dataset(self, split, transform_train=None, transform_test=None):
        print('==> Preparing CIFAR 10 data..')

        assert split in ['train', 'test']
        if split == 'test':
            ds = datasets.CIFAR10(root=self.data_path, train=False, download=False, transform=self.transform_test)
        else:
            ds = datasets.CIFAR10(root=self.data_path, train=True, download=False, transform=self.transform_train)

        return ds

class CIFAR100:
    """
    CIFAR-100 dataset
    """
    NUM_CLASSES = 100
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    SIZE = 32
    SOBEL_UPSAMPLE_SIZE = 64

    def __init__(self, data_path):
        self.data_path = data_path
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # Random crop with padding for augmentation
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=CIFAR100.MEAN,std=CIFAR100.STD)
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR100.MEAN,std=CIFAR100.STD)
        ])

    def get_dataset(self, split, transform_train, transform_test):
        print('==> Preparing CIFAR 100 data..')

        assert split in ['train', 'test']
        if split == 'test':
            ds = datasets.CIFAR100(root=self.data_path, train=False, download=False, transform=self.transform_test)
        else:
            ds = datasets.CIFAR100(root=self.data_path, train=True, download=False, transform=self.transform_train)

        return ds

class CelebA:
    NUM_CLASSES = 2
    CLASS_NAMES = ['female', 'male']
    CLASS_ID = {0: "female", 1: "male"}
    MEAN = [0, 0, 0]
    STD = [1, 1, 1, ]
    SIZE = 96

    def __init__(self, data_path, **kwargs):
        self.data_path = data_path
        self.arch = kwargs.get('arch', None)

        if 'clip' in self.arch:
            self.transform_train = transforms.Compose(
                [transforms.Resize((224, 224)),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=CelebA.MEAN, std=CelebA.STD)
                 ])
            self.transform_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=CelebA.MEAN,std=CelebA.STD)
            ])
        else:
            self.transform_train = transforms.Compose([
                    transforms.Resize((CelebA.SIZE, CelebA.SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=CelebA.MEAN,std=CelebA.STD),
                ])
            self.transform_test = transforms.Compose([
                    transforms.Resize((CelebA.SIZE, CelebA.SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=CelebA.MEAN,std=CelebA.STD),
                ])

    def get_dataset(self, split, transform_train=None, transform_test=None,unlabel_skew=True):
        assert split in ['train', 'val', 'test', 'unlabeled']

        if split == 'test':
            ds = datasets.CelebA(root=self.data_path, split='test', transform=self.transform_test)
        # elif split == 'val':
        #     ds = torchvision.norm_datasets.CelebA(root=self.data_path, split='valid', transform=trans)
        else:
            ds = datasets.CelebA(root=self.data_path, split='valid', transform=self.transform_train)
        attr_names = ds.attr_names
        attr_names_map = {a: i for i, a in enumerate(attr_names)}
        ds = celeb_indicies(split, ds, attr_names_map, unlabel_skew)

        return ds

class CIFAR10_Imb:
    """
    CIFAR-10 dataset
    """
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    CLASS_ID = {0: "car (automobile)", 1: "airplane", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                6: "frog", 7: "horse", 8: "cargo ship", 9: "truck"}
    NUM_CLASSES = 10
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    SIZE = 32

    def __init__(self, data_path, perc=1.0, c_gamma=-1, corrupt_prob=0.0):
        self.data_path = data_path
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # Random crop with padding for augmentation
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=CIFAR10.MEAN,std=CIFAR10.STD)
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10.MEAN,std=CIFAR10.STD)
        ])

        self.perc = perc
        self.c_gamma = c_gamma
        self.corrupt_prob = corrupt_prob

    def get_dataset(self, split, transform_train, transform_test):
        print('==> Preparing CIFAR 10 data..')

        assert split in ['train', 'test']
        if split == 'test':
            ds = CIFAR10ImbalancedNoisy(corrupt_prob=0.0, gamma=-1, n_min=250, n_max=5000, num_classes=10, perc=1.0, root=self.data_path, train=False, download=True, transform=self.transform_test, )
        else:
            ds = CIFAR10ImbalancedNoisy(corrupt_prob=self.corrupt_prob, gamma=self.c_gamma, n_min=250, n_max=5000, num_classes=10, perc=self.perc, root=self.data_path, train=True, download=True, transform=self.transform_train)

        return ds

class CIFARTint():
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    CLASS_ID = {0: "car (automobile)", 1: "airplane", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                6: "frog", 7: "horse", 8: "cargo ship", 9: "truck"}
    NUM_CLASSES = 10
    MEAN = [0.4192, 0.4124, 0.3804]
    STD = [0.2714, 0.2679, 0.2771]
    SIZE = 96

    def __init__(self, data_path, **kwargs):
        self.data_path = data_path
        self.cif_dataset = CIFAR10(self.data_path, **kwargs)
        self.arch = kwargs.get('arch', None)

        if 'clip' in self.arch:
            self.transform_train = transforms.Compose(
                [transforms.ToPILImage(),
                 # transforms.Resize((224, 224)),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=CIFAR10.MEAN, std=CIFAR10.STD)
                 ])
            self.transform_test = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10.MEAN,std=CIFAR10.STD)
            ])
        else:
            self.transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),  # Random crop with padding for augmentation
                transforms.RandomHorizontalFlip(),  # Random horizontal flip
                transforms.ToTensor(),  # Convert image to tensor
                transforms.Normalize(mean=CIFAR10.MEAN,std=CIFAR10.STD)
            ])
            self.transform_test = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10.MEAN,std=CIFAR10.STD)
            ])
    def get_dataset(self, split, transform_train, transform_test):
        dataset = self.cif_dataset.get_dataset(split)
        if split == 'train':
            ds = cif_tint(dataset, split=split, transform=transform_train)
        else:
            ds = cif_tint(dataset, split=split, transform=transform_test)

        return ds

class TinyImagenet():
    NUM_CLASSES = 200
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64

    def __init__(self, data_path):
        self.data_path = data_path
        self.transform_train = transforms.Compose(
            [transforms.RandomCrop(64, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
            transforms.Normalize(mean=TinyImagenet.MEAN,std=TinyImagenet.STD),
            ])
        self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=TinyImagenet.MEAN,std=TinyImagenet.STD),
            ])

    def get_dataset(self, split, transform_train=None, transform_test=None):
        assert split in ['train', 'test']
        if split == 'test':
            # ds = TinyImageNetValWithLabels(root=self.data_path, annotations_file=os.path.join(self.data_path, 'val', 'val_annotations.txt'),transform=self.transform_test)
            ds = torchvision.datasets.ImageFolder(root=os.path.join(self.data_path, 'val'), transform=self.transform_test)
        else:
            ds = torchvision.datasets.ImageFolder(root=os.path.join(self.data_path, 'train'), transform=self.transform_train)

        self.CLASS_ID = ds.class_to_idx
        # with open('/volumes1/datasets/tiny-imagenet-200/tiny-class-id', 'w') as f:
        #     json.dump(ds.class_to_idx, f)

        return ds

class TinyImagenetStyle():
    NUM_CLASSES = 200
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64

    def __init__(self, data_path, alpha):
        self.data_path = data_path
        self.alpha = str(alpha)
        self.transform_test = transforms.Compose([
            transforms.Resize((TinyImagenetStyle.SIZE, TinyImagenetStyle.SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=TinyImagenetStyle.MEAN,std=TinyImagenetStyle.STD),
            ])

    def get_dataset(self):
        ds = torchvision.datasets.ImageFolder(root=os.path.join(self.data_path, self.alpha), transform=self.transform_test)
        return ds

class Imagenet_R():
    NUM_CLASSES = 1000
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64
    def __init__(self, data_path, mapping_file):
        self.data_path = data_path
        with open(mapping_file, 'r') as f:
            self.label_mapping = json.load(f)
        self.transform_test= transforms.Compose([
                transforms.Resize((Imagenet_R.SIZE,Imagenet_R.SIZE)),
                # transforms.CenterCrop(56),
                transforms.ToTensor(),
                transforms.Normalize(mean=Imagenet_R.MEAN,std=Imagenet_R.STD),
            ])
    def get_dataset(self, split=None, transform_train=None, transform_test=None):
        ds = MappedImageFolder(root=os.path.join(self.data_path), transform=self.transform_test,
                               label_mapping=self.label_mapping)
        return ds

class Imagenet_O():
    NUM_CLASSES = 1000
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64
    def __init__(self, data_path, mapping_file):
        self.data_path = data_path
        with open(mapping_file, 'r') as f:
            self.label_mapping = json.load(f)
        self.transform_test= transforms.Compose([
                transforms.Resize((Imagenet_O.SIZE,Imagenet_O.SIZE)),
                # transforms.CenterCrop(56),
                transforms.ToTensor(),
                transforms.Normalize(mean=Imagenet_O.MEAN,std=Imagenet_O.STD),
            ])
    def get_dataset(self, split=None, transform_train=None, transform_test=None):
        ds = MappedImageFolder(root=os.path.join(self.data_path), transform=self.transform_test,
                               label_mapping=self.label_mapping)
        return ds

class Imagenet_A():
    NUM_CLASSES = 1000
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64
    def __init__(self, data_path, mapping_file):
        self.data_path = data_path
        with open(mapping_file, 'r') as f:
            self.label_mapping = json.load(f)
        self.transform_test= transforms.Compose([
                transforms.Resize(((Imagenet_A.SIZE,Imagenet_A.SIZE))),
                # transforms.CenterCrop(56),
                transforms.ToTensor(),
                transforms.Normalize(mean=Imagenet_A.MEAN,std=Imagenet_A.STD),
            ])
    def get_dataset(self, split=None, transform_train=None, transform_test=None):
        ds = MappedImageFolder(root=os.path.join(self.data_path), transform=self.transform_test,
                               label_mapping=self.label_mapping)
        return ds

class Imagenet100():
    NUM_CLASSES = 100
    MEAN = [0.48145466, 0.4578275, 0.40821073]
    STD = [0.26862954, 0.26130258, 0.27577711]
    SIZE = 224

    def __init__(self, data_path):
        self.data_path = data_path
        self.transform_train = transforms.Compose(
            [transforms.Resize((Imagenet100.SIZE, Imagenet100.SIZE)),
             transforms.ToTensor(),
            transforms.Normalize(mean=Imagenet100.MEAN,std=Imagenet100.STD),
            ])
        self.transform_test = transforms.Compose([
                transforms.Resize((Imagenet100.SIZE, Imagenet100.SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=Imagenet100.MEAN,std=Imagenet100.STD),
            ])
        with open(os.path.join(self.data_path, "Labels.json"), 'r') as f:
            labels = json.load(f)
        self.class_to_idx_unified = {class_name: idx for idx, class_name in enumerate(labels.keys())}

    def get_dataset(self, split, transform_train=None, transform_test=None):
        assert split in ['train', 'test']

        if split == 'train':
            # train_datasets = []
            # train_subdirs = [os.path.join(self.data_path, f'train.X{i}') for i in range(1, 5)]
            # for subdir in train_subdirs:
            #     dataset = torchvision.datasets.ImageFolder(subdir, transform=transform_train)
            #     dataset.class_to_idx = self.class_to_idx_unified
            #     dataset.classes = list(self.class_to_idx_unified.keys())
            #     train_datasets.append(dataset)
            # ds = ConcatDataset(train_datasets)
            ds = torchvision.datasets.ImageFolder(root=os.path.join(self.data_path, 'train'), transform=self.transform_train)
        else:
            ds = torchvision.datasets.ImageFolder(os.path.join(self.data_path, 'val.X'), transform=self.transform_test)

        self.CLASS_ID = ds.class_to_idx
        # with open('/volumes1/datasets/tiny-imagenet-200/tiny-class-id', 'w') as f:
        #     json.dump(ds.class_to_idx, f)

        return ds

class Waterbirds:
    NUM_CLASSES = 2
    SIZE = 96
    CLASS_NAMES = ['landbird', 'waterbird']
    CLASS_ID = {0: "landbird", 1: "waterbird"}
    def __init__(self, data_path, **kwargs):
        self.data = pd.read_csv(os.path.join(data_path, "metadata.csv"))
        self.birds = ["landbird", "waterbird"]
        self.background = ["land", "water"]
        self.data_dir = data_path
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((Waterbirds.SIZE, Waterbirds.SIZE)),
            # transforms.Normalize(mean=CelebA.MEAN, std=CelebA.STD),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((Waterbirds.SIZE, Waterbirds.SIZE)),
            # transforms.Normalize(mean=CelebA.MEAN, std=CelebA.STD),
        ])

        self.train_data = self.data.loc[self.data["split"] == 0]
        self.test_data = self.data.loc[self.data["split"] == 1]

    def get_dataset(self, split, transform_train=None, transform_test=None, seed=0):
        train_data = self.data.sample(frac=0.8, random_state=seed)
        val_test_data = self.data.drop(train_data.index)
        val_data = val_test_data.sample(frac=0.5, random_state=seed)
        test_data = val_test_data.drop(val_data.index)
        if split == "train":
            self.data = train_data.reset_index(drop=True)
        elif split == "val":
            self.data = val_data.reset_index(drop=True)
        elif split == "test":
            self.data = test_data.reset_index(drop=True)
        else:
            raise ValueError("split does not exist")

        if split == 'train':
            ds = waterbird_shortcut(self.data_dir, self.train_data, transform=self.transform_train)
        else:
            ds = waterbird_shortcut(self.data_dir, self.test_data, transform=self.transform_test)
        return ds


DATASETS = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'tinyimagenet': TinyImagenet,
    'celeba' : CelebA,
    'cifar10_imb': CIFAR10_Imb,
    'cifartint': CIFARTint,
    'imagenet_r': Imagenet_R,
    'imagenet_o': Imagenet_O,
    'imagenet_a': Imagenet_A,
    'tinystyle':TinyImagenetStyle,
    'imagenet100':Imagenet100,
    'waterbirds': Waterbirds,
    # 'col_mnist': coloredMNIST,
    # 'cor_cifar10': Corrupt_CIFAR10,
    # 'cor_tinyimagenet':Corrupt_TinyImagenet
}

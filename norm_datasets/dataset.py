import os
import torchvision
from torchvision import datasets, transforms
from norm_datasets.utils import celeb_indicies, CelebA_Wrapper
from norm_datasets.cifar_imbalance import CIFAR10ImbalancedNoisy
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

    def __init__(self, data_path):
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

    def get_dataset(self, split, transform_train, transform_test):
        print('==> Preparing CIFAR 10 data..')

        assert split in ['train', 'test']
        if split == 'test':
            ds = datasets.CIFAR10(root=self.data_path, train=False, download=True, transform=self.transform_test)
        else:
            ds = datasets.CIFAR10(root=self.data_path, train=True, download=True, transform=self.transform_train)

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

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        print('==> Preparing CIFAR 100 data..')

        assert split in ['train', 'test']
        if split == 'test':

            ds = datasets.CIFAR100(root=self.data_path, train=False, download=True, transform=transform_test)
        else:
            ds = datasets.CIFAR10(root=self.data_path, train=True, download=True, transform=transform_train)

        return ds
# class TinyImagenet():
#     NUM_CLASSES = 200
#     MEAN = [0.4802, 0.4481, 0.3975]
#     STD = [0.2302, 0.2265, 0.2262]
#     SIZE = 64
#     SOBEL_UPSAMPLE_SIZE = 128
#
#     def __init__(self, data_path):
#         self.data_path = data_path
#
#     def get_dataset(self, split, transform_train, transform_test):
#         assert split in ['train', 'test']
#         if split == 'test':
#             ds = ImageFilelist(root=self.data_path, flist=os.path.join(self.data_path, "val_kv_list.txt"), transform=transform_test,)
#         else:
#             ds = ImageFilelist(root=self.data_path, flist=os.path.join(self.data_path, "train_kv_list.txt"), transform=transform_train)
#
#         return ds

class CelebA:
    NUM_CLASSES = 2
    CLASS_NAMES = ['female', 'male']
    CLASS_ID = {0: "female", 1: "male"}
    MEAN = [0, 0, 0]
    STD = [1, 1, 1, ]
    SIZE = 96

    def __init__(self, data_path):
        self.data_path = data_path
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

    def get_dataset(self, split, transform_train, transform_test,unlabel_skew=True):
        assert split in ['train', 'val', 'test', 'unlabeled']

        if split == 'test':
            ds = datasets.CelebA(root=self.data_path, split='test', transform=transform_test)
        # elif split == 'val':
        #     ds = torchvision.norm_datasets.CelebA(root=self.data_path, split='valid', transform=trans)
        else:
            ds = datasets.CelebA(root=self.data_path, split='valid', transform=transform_train)
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

    def __init__(self, data_path, perc=1.0, gamma=-1, corrupt_prob=0.0):
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
        self.gamma = gamma
        self.corrupt_prob = corrupt_prob

    def get_dataset(self, split, transform_train, transform_test):
        print('==> Preparing CIFAR 10 data..')

        assert split in ['train', 'test']
        if split == 'test':
            ds = CIFAR10ImbalancedNoisy(corrupt_prob=0.0, gamma=-1, n_min=250, n_max=5000, num_classes=10, perc=1.0, root=self.data_path, train=False, download=True, transform=self.transform_test, )
        else:
            ds = CIFAR10ImbalancedNoisy(corrupt_prob=self.corrupt_prob, gamma=self.gamma, n_min=250, n_max=5000, num_classes=10, perc=self.perc, root=self.data_path, train=True, download=True, transform=self.transform_train)

        return ds


DATASETS = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    # 'tinyimagenet': TinyImagenet,
    'celeba' : CelebA,
    'cifar10_imb': CIFAR10_Imb,
    # 'col_mnist': coloredMNIST,
    # 'cor_cifar10': Corrupt_CIFAR10,
    # 'cor_tinyimagenet':Corrupt_TinyImagenet

}

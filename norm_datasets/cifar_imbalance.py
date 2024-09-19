"""
cifar-10 dataset, with support for random labels
"""
import numpy as np
from copy import deepcopy
from torchvision import datasets


class CIFAR10ImbalancedNoisy(datasets.CIFAR10):
    """CIFAR10 dataset, with support for Imbalanced and randomly corrupt labels.

    Params
    ------
    corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
    num_classes: int
    Default 10. The number of classes in the dataset.
    """
    def __init__(self, corrupt_prob=0.0, gamma=-1, n_min=250, n_max=5000, num_classes=10, perc=1.0, **kwargs):
        super(CIFAR10ImbalancedNoisy, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.perc = perc
        self.gamma = gamma
        self.n_min = n_min
        self.n_max = n_max
        self.true_labels = deepcopy(self.targets)

        if perc < 1.0:
            print('*' * 30)
            print('Creating a Subset of Dataset')
            self.get_subset()
            (unique, counts) = np.unique(self.targets, return_counts=True)
            frequencies = np.asarray((unique, counts)).T
            print(frequencies)

        if gamma > 0:
            print('*' * 30)
            print('Creating Imbalanced Dataset')
            self.imbalanced_dataset()
            self.true_labels = deepcopy(self.targets)

        if corrupt_prob > 0:
            print('*' * 30)
            print('Applying Label Corruption')
            self.corrupt_labels(corrupt_prob)

    def get_subset(self):
        np.random.seed(12345)

        lst_data = []
        lst_targets = []
        targets = np.array(self.targets)
        for class_idx in range(self.num_classes):
            class_indices = np.where(targets == class_idx)[0]
            num_samples = int(self.perc * len(class_indices))
            sel_class_indices = class_indices[:num_samples]
            lst_data.append(self.data[sel_class_indices])
            lst_targets.append(targets[sel_class_indices])

        self.data = np.concatenate(lst_data)
        self.targets = np.concatenate(lst_targets)

        assert len(self.targets) == len(self.data)

    def imbalanced_dataset(self):
        np.random.seed(12345)
        X = np.array([[1, -self.n_max], [1, -self.n_min]])
        Y = np.array([self.n_max, self.n_min * self.num_classes ** (self.gamma)])

        a, b = np.linalg.solve(X, Y)

        classes = list(range(1, self.num_classes + 1))

        imbal_class_counts = []
        for c in classes:
          num_c = int(np.round(a / (b + (c) ** (self.gamma))))
          # print(c, num_c)
          print(num_c)
          imbal_class_counts.append(num_c)

        targets = np.array(self.targets)

        # Get class indices
        class_indices = [np.where(targets == i)[0] for i in range(self.num_classes)]

        # Get imbalanced number of instances
        imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
        imbal_class_indices = np.hstack(imbal_class_indices)

        np.random.shuffle(imbal_class_indices)

        # Set target and data to dataset
        self.targets = targets[imbal_class_indices]
        self.data = self.data[imbal_class_indices]

        assert len(self.targets) == len(self.data)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.num_classes, mask.sum())
        labels[mask] = rnd_labels

        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]
        self.targets = labels


class CIFAR10RandomSubset(datasets.CIFAR10):
    """CIFAR10 dataset, with support for randomly corrupt labels.

    Params
    ------
    corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
    num_classes: int
    Default 10. The number of classes in the dataset.
    """
    def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
        super(CIFAR10RandomSubset, self).__init__(**kwargs)
        self.n_classes = num_classes

        labels = np.array(self.targets)
        self.data = self.data[labels < num_classes]
        labels = labels[labels < num_classes]
        labels = [int(x) for x in labels]
        self.targets = labels

        print('Number of Training Samples:', len(self.targets))

        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):

        labels = np.array(self.targets)

        print('Original Labels:', labels)

        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels

        print('Noisy Labels:', labels)

        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]
        self.targets = labels

# =============================================================================
# Test Bench
# =============================================================================

# perc = 0.8
#
# dataset = CIFAR10ImbalancedNoisy(
#     root='./data/',
#     train=True,
#     download=True,
#     num_classes=10,
#     perc=perc,
#     gamma=1,
#     n_max=int(5000 * perc),
#     n_min=int(250 * perc),
#     corrupt_prob=0,
# )
#

# dataset = CIFAR10Subset(perc=1, root='data')
# (unique, counts) = np.unique(dataset.targets, return_counts=True)
# frequencies = np.asarray((unique, counts)).T
# print(frequencies)


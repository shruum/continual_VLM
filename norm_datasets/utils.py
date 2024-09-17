import os
import torch
import os.path
import PIL
import numpy as np
from PIL import Image
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg, download_file_from_google_drive
from typing import Union
from functools import partial
import torch.utils.data as data
from typing import Any, Callable, List, Optional, Tuple
import torch.utils.data as data_utils

class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(
            self,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            transform_fp: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.transform_fp = transform_fp
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self) -> str:
        return ""

class CelebA_Wrapper(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                ``identity`` (int): label for each person (data points with the same identity are the same person)
                ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                    righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)
            Defaults to ``attr``. If empty, ``None`` will be returned as target.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            transform_fp: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = True,
    ) -> None:
        import pandas
        super(CelebA_Wrapper, self).__init__(root, transform=transform,
                                             transform_fp=transform_fp,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None) if split_ is None else (splits[1] == split_)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self) -> None:
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        img_fp = X

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

def celeb_indicies(split, ds, attr_names_map, unlabel_skew=True):
    male_mask = ds.attr[:, attr_names_map['Male']] == 1
    female_mask = ds.attr[:, attr_names_map['Male']] == 0
    blond_mask = ds.attr[:, attr_names_map['Blond_Hair']] == 1
    not_blond_mask = ds.attr[:, attr_names_map['Blond_Hair']] == 0

    indices = torch.arange(len(ds))

    if split == 'train':
        male_blond = indices[torch.logical_and(male_mask, blond_mask)]
        male_not_blond = indices[torch.logical_and(male_mask, not_blond_mask)]
        female_blond = indices[torch.logical_and(female_mask, blond_mask)]
        female_not_blond = indices[torch.logical_and(female_mask, not_blond_mask)]
        p_male_blond = len(male_blond) / float(len(indices))
        p_male_not_blond = len(male_not_blond) / float(len(indices))
        p_female_blond = len(female_blond) / float(len(indices))
        p_female_not_blond = len(female_not_blond) / float(len(indices))

        # training set must have 500 male_not_blond and 500 female_blond
        train_N = 1000
        training_male_not_blond = male_not_blond[:train_N]
        training_female_blond = female_blond[:train_N]

        unlabeled_male_not_blond = male_not_blond[train_N:]
        unlabeled_female_blond = female_blond[train_N:]
        unlabeled_male_blond = male_blond
        unlabeled_female_not_blond = female_not_blond

        if unlabel_skew:
            # take 1000 from each category
            unlabeled_N = 1000
            unlabeled_male_not_blond = unlabeled_male_not_blond[:unlabeled_N]
            unlabeled_female_blond = unlabeled_female_blond[:unlabeled_N]
            unlabeled_male_blond = unlabeled_male_blond[:unlabeled_N]
            unlabeled_female_not_blond = unlabeled_female_not_blond[:unlabeled_N]
        else:
            total_N = 4000
            extra = total_N - int(p_male_not_blond * total_N) - int(p_female_blond * total_N) - int(
                p_male_blond * total_N) - int(p_female_not_blond * total_N)
            unlabeled_male_not_blond = unlabeled_male_not_blond[:int(p_male_not_blond * total_N)]
            unlabeled_female_blond = unlabeled_female_blond[:int(p_female_blond * total_N)]
            unlabeled_male_blond = unlabeled_male_blond[:int(p_male_blond * total_N)]
            unlabeled_female_not_blond = unlabeled_female_not_blond[:(int(p_female_not_blond * total_N) + extra)]

        train_indices = np.concatenate([training_male_not_blond, training_female_blond])
        unlabelled_indices = np.concatenate(
            [unlabeled_male_not_blond, unlabeled_female_blond, unlabeled_male_blond, unlabeled_female_not_blond])
        for index in unlabelled_indices:
            assert index not in train_indices

        if split == 'train':
            indices = train_indices
        else:
            indices = unlabelled_indices

    imgs = []
    imgs_fp = []
    ys = []
    metas = []
    is_blonde= []
    for i in indices:

        img, attr = ds[i]

        imgs.append(img)
        ys.append(attr[attr_names_map['Male']])
        is_blonde.append(blond_mask[i])
        if male_mask[i]:
            agree = False if blond_mask[i] else True
        else:
            agree = True if blond_mask[i] else False
        metas.append({'agrees': agree, 'blond': blond_mask[i], 'male': male_mask[i]})

    print(split, len(indices))
    if split == 'test':
        return data_utils.TensorDataset(torch.stack(imgs), torch.tensor(ys))#, torch.tensor(is_blonde))
    else:
        return data_utils.TensorDataset(torch.stack(imgs), torch.tensor(ys))

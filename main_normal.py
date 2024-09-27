# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy # needed (don't change it)
import importlib
import os
import sys
import socket

mammoth_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(mammoth_path)
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/cl_datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')

from argparse import ArgumentParser
from utils.args import add_management_args, add_experiment_args, add_auxiliary_args
from cl_datasets import ContinualDataset
from utils.best_args import best_args
from utils.conf import set_random_seed
from backbone.ResNet18 import *
from backbone.ResNet_mam_llm import resnet18mamllm
from backbone.ResNet_mam import *
import torch
import uuid
import datetime
from norm_datasets.dataset import DATASETS
from utils.normal_training import train_normal
from models.normal import Normal
import clip

def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)

    torch.set_num_threads(4)
    add_management_args(parser)
    add_experiment_args(parser)
    add_auxiliary_args(parser)

    args = parser.parse_args()
    if args.seed is not None:
        set_random_seed(args.seed)

    torch.set_num_threads(4)
    args.num_workers = 4
    return args


def main_normal(args=None):
    lecun_fix()
    if args is None:
        args = parse_args()

    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE IS {device}")
    # If CUDA is available, print detailed GPU information
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
            print(f"  Current Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
            print(f"  Current Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
    else:
        print("CUDA is not available. Using CPU.")

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    if args.dataset == 'cifar10_imb':
        dataset = DATASETS[args.dataset](args.dataset_dir, args.perc, args.gamma, args.corrupt_prob)
    else:
        dataset = DATASETS[args.dataset](args.dataset_dir)
    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()

    cifar_resnet = True
    if args.llama:
        backbone = resnet18mamllm(dataset.NUM_CLASSES, 64, args.llm_block).to(device)
    else:
        backbone = resnet18mam(dataset.NUM_CLASSES).to(device)
    model = Normal(args, backbone, dataset, device)

    if args.debug_mode:
        args.nowand = 1

    # set job name
    # setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))
    train_normal(args, dataset, model)



if __name__ == '__main__':
    main_normal()

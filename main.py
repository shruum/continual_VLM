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

from cl_datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args, add_gcil_args, add_av_dataset_args
from cl_datasets import ContinualDataset
from utils.continual_training import train as ctrain
from cl_datasets import get_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
# import setproctitle
import torch
import uuid
import datetime
from pathlib import Path


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--arch', type=str, required=True,
                        help='Arch name.')# choices=('resnet18','resnet50'))
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--dataset_dir', type=str, default='data',
                        help='Base directory for cl_datasets.')
    parser.add_argument('--output_dir', type=str, default='experiments_res50',
                        help='Base directory for logging results.')
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')

    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        if args.dataset in ['gcil-cifar100']:
            add_gcil_args(parser)
        if args.dataset in ['seq_vggsound']:
            add_av_dataset_args(parser)
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if hasattr(mod, 'Buffer'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        to_parse = sys.argv[1:] + ['--' + k + '=' + str(v) for k, v in best.items()]
        to_parse.remove('--load_best_args')
        args = parser.parse_args(to_parse)
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        if args.dataset in ['gcil-cifar100']:
            add_gcil_args(parser)
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    return args


def main(args=None):
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
    dataset = get_dataset(args)

    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
    if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and args.minibatch_size is None:
        args.minibatch_size = dataset.get_minibatch_size()

    backbone = dataset.get_backbone()
    if args.llama:
        print("Loading LLaMA checkpoints")
        checkpoints = sorted(Path(args.llama_path).glob("*.pth"))
        print(checkpoints)
        ckpt_path = checkpoints[0]
        print(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu") #
        backbone.llama.custom_load_state_dict(checkpoint, tail=True, strict=True) #.to(device)
        backbone.llama.to(device)

    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    if args.debug_mode:
        args.nowand = 1

    # set job name
    # setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))

    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args)
    else:
        # assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        if "general-continual" in model.COMPATIBILITY:
            ctrain(args)


if __name__ == '__main__':
    main()

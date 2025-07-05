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
from backbone.ResNet import *
from backbone.ResNet_llm import *
from backbone.ResNet_mam_llm import *
from backbone.vit import *
from backbone.vit_llm import *
from backbone.ResNet_mam import *
from backbone.clip_classifier import ClipClassifier
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
        dataset = DATASETS[args.dataset](args.dataset_dir, args.perc, args.c_gamma, args.corrupt_prob)
    else:
        if args.dataset == 'cifar10' or args.dataset == 'celeba' or args.dataset == 'cifartint':
            dataset_args = {"data_path": args.dataset_dir, "arch": args.arch}
        else:
            dataset_args = {"data_path": args.dataset_dir}
        dataset = DATASETS[args.dataset](**dataset_args)
    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()

    cifar_resnet = True
    # if args.llama:
    #     backbone = resnet18mamllm(dataset.NUM_CLASSES, 64, args.llm_block).to(device)
    if args.arch == 'clip_vit':
        # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model, preprocess = clip.load("ViT-B/32", device=device)
        model = model.float()
        # Freeze CLIP's image encoder
        for param in model.visual.parameters():
            param.requires_grad = False

        feature_dim = model.visual.output_dim
        backbone = ClipClassifier(model.visual, feature_dim, dataset.NUM_CLASSES).to(device)
    elif args.arch == 'clip_res50':
        model, preprocess = clip.load("RN50", device=device)
        model = model.float()
        for param in model.visual.parameters():
            param.requires_grad = False
        feature_dim = model.visual.output_dim
        backbone = ClipClassifier(model.visual, feature_dim, dataset.NUM_CLASSES).to(device)

    elif args.arch == "vitclip":
        backbone = vitclip(dataset.NUM_CLASSES).to(device)
    elif args.arch == 'vitsmall':
        backbone = vitsmall(dataset.NUM_CLASSES).to(device)
    elif args.arch == "vitsmallllm":
        backbone = vitsmallllm(dataset.NUM_CLASSES, args.llm_block).to(device)
    elif args.arch == "resnet18mamllm":
        backbone = resnet18mamllm(dataset.NUM_CLASSES, 64, args.llm_block, args.llm_pretrain).to(device)
    elif args.arch == "resnet50":
        backbone = resnet50(dataset.NUM_CLASSES).to(device)
    elif args.arch == "resnet18":
        backbone = resnet18(dataset.NUM_CLASSES).to(device)
    elif args.arch == "resnet18llm":
        backbone = resnet18llm(dataset.NUM_CLASSES, 64, args.llm_block).to(device)
    elif args.arch == "resnet50llm":
        backbone = resnet50llm(dataset.NUM_CLASSES, 64, args.llm_block).to(device)
    elif args.arch == "resnet18mam":
        backbone = resnet18mam(dataset.NUM_CLASSES).to(device)
    elif args.arch == "resnet50mam":
        backbone = resnet50mam(dataset.NUM_CLASSES).to(device)
    elif args.arch == "resnet50mamllm":
        backbone = resnet50mamllm(dataset.NUM_CLASSES).to(device)
    else:
        raise ValueError('Backbone not found')
    print("Loading backbone {}".format(backbone._get_name()))
    model = Normal(args, backbone, dataset, device)

    if args.debug_mode:
        args.nowand = 1

    # set job name
    # setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))
    train_normal(args, dataset, model)



if __name__ == '__main__':
    main_normal()

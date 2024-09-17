# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from cl_datasets import NAMES as DATASET_NAMES
from models import get_all_models
from datetime import datetime


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--experiment_id', type=str,
                        default=datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                        help='Unique identifier for the experiment.')
    parser.add_argument('--dataset_dir', type=str, default='data',
                        help='Base directory for datasets.')
    parser.add_argument('--output_dir', type=str, default='experiments',
                        help='Base directory for logging results.')
    parser.add_argument('--dataset', type=str, required=True,
                        # choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--mnist_seed', type=int, default=0,
                        help='Seed for reproducible MNIST variants.')
    parser.add_argument('--n_tasks_cif', type=int, default=5,
                        help='Number of tasks in CIFAR-100.')
    parser.add_argument('--n_tasks_mnist', type=int, default=20,
                        help='Number of tasks in Rotated/Permuted MNIST.')
    parser.add_argument('--deg_inc', type=float, default=8,
                        help='Deg increment for fixed rotated MNIST')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--arch', type=str, required=True,
                        help='Arch name.') #choices=('resnet18','resnet50'))

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--scheduler', default='None', choices=['None', 'cosine', 'multistep'])
    parser.add_argument('--n_epochs', type=int,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--llama', default=False)
    parser.add_argument('--llama_path', default='/volumes1/llama/llama-models/models/llama3_1/Meta-Llama-3.1-8B', type=str)
    parser.add_argument('--mode', default='normal', choices=['normal', 'vlm', 'madry', 'trades'])


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None, help='The random seed.')
    parser.add_argument('--notes', type=str, default=None, help='Notes for this run.')

    parser.add_argument('--non_verbose', default=0, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Enable csv logging')
    parser.add_argument('--tensorboard', default=0, choices=[0, 1], type=int, help='Enable tensorboard logging')

    parser.add_argument('--validation', default=0, choices=[0, 1], type=int, help='Test on the validation set')
    parser.add_argument('--ignore_other_metrics', default=0, choices=[0, 1], type=int, help='disable additional metrics')
    parser.add_argument('--debug_mode', type=int, default=0, help='Run only a few forward steps per epoch')
    parser.add_argument('--nowand', default=0, choices=[0, 1], type=int, help='Inhibit wandb logging')
    parser.add_argument('--wandb_entity', type=str, default='regaz', help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='mammoth', help='Wandb project name')
    parser.add_argument('--save_model', action='store_true')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True, help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, help='The batch size of the memory buffer.')

def add_auxiliary_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--aux', type=str, default='shape',
                        help='The type of auxiliary data')
    parser.add_argument('--img_size', type=int, default=32,
                        help='Image size.')
    parser.add_argument('--loss_type', nargs='*', type=str, default=['l2'], help="--loss_type kl at")
    parser.add_argument('--loss_wt', nargs='*', type=float, default=[1.0, 1.0, 1.0, 1.0])
    parser.add_argument('--dir_aux', action='store_true')
    parser.add_argument('--buf_aux', action='store_true')
    parser.add_argument('--rev_proj', action='store_true')
    parser.add_argument('--ser', action='store_true')
    parser.add_argument('--aug_prob', default=0.0, type=float)
    parser.add_argument('--data_combine', action='store_true')
    parser.add_argument('--loss_mode', type=str, default='l2')
    parser.add_argument('--text_model', type=str, required=False)
    parser.add_argument('--ser_weight', type=float, default=0.1)
    parser.add_argument('--gpt_path', type=str, required=False)
    # choices=["cl_datasets/metadata/cifar10_descriptions.json",
    # "/volumes1/datasets/tinyimagenet_description.json",
    # "/volumes1/datasets/domainnet_description_100.json",
    # "cl_datasets/metadata/lvis_gpt3_text-davinci-002_descriptions_author.json",
    # "/volumes1/datasets/celeba_description.json"])

def add_gcil_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments required for GCIL-CIFAR100 Dataset.
    :param parser: the parser instance
    """
    # arguments for GCIL-CIFAR100
    parser.add_argument('--gil_seed', type=int, default=1993, help='Seed value for GIL-CIFAR task sampling')
    parser.add_argument('--pretrain', action='store_true', default=False, help='whether to use pretrain')
    parser.add_argument('--phase_class_upper', default=50, type=int, help='the maximum number of classes')
    parser.add_argument('--epoch_size', default=1000, type=int, help='Number of samples in one epoch')
    parser.add_argument('--pretrain_class_nb', default=0, type=int, help='the number of classes in first group')
    parser.add_argument('--weight_dist', default='unif', type=str, choices=['unif', 'longtail'],
                        help='what type of weight distribution assigned to classes to sample (unif or longtail)')


def add_av_dataset_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments required for Audio Video Datasets.
    :param parser: the parser instance
    """
    # arguments for VGGSound
    parser.add_argument('--vggsound_csv_path', type=str, default=r'/volumes2/workspace/nie_continual_learning/mammoth-multimodal/datasets/utils/vggsound/vggsound.csv')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second for the videos')
    parser.add_argument('--num_video_frames', type=int, default=4, help='Number of frames to extract for each video')
    parser.add_argument('--fusion_method', type=str, default='film', help='Method for fusing the modalities')
    parser.add_argument('--modalities_used', type=str, default='audio_video', help='Method for fusing the modalities')
    # Arguments for Domain-IL
    parser.add_argument('--vggsound_categories_path', type=str, default=r'/volumes2/workspace/multimodal_cl/nie_continual_learning/mammoth-multimodal/supercategories_sorted.csv')
    parser.add_argument('--domain_vgg_ntasks', type=int, default=5)
    parser.add_argument('--domain_vgg_seed', type=int, default=0)
    # Arguments for GCIL
    parser.add_argument('--gcil_seed', type=int, default=1992)
    parser.add_argument('--pretrain', action='store_true', default=False, help='whether to use pretrain')
    parser.add_argument('--phase_class_upper', default=50, type=int, help='the maximum number of classes')
    parser.add_argument('--epoch_size', default=1000, type=int, help='Number of samples in one epoch')
    parser.add_argument('--pretrain_class_nb', default=0, type=int, help='the number of classes in first group')
    parser.add_argument('--weight_dist', default='unif', type=str, choices=['unif', 'longtail'],
                        help='what type of weight distribution assigned to classes to sample (unif or longtail)')


def add_transf_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--patch_embed_type', type=str, default='Patch',
                        help='Type of patch embedding for ViT ["Conv", "Patch"]')
    parser.add_argument('--aug_version', type=str, default='v1',
                        help='Type of Augmentations ["v1", "v2", "v3"]')
    parser.add_argument('--diff_lr', action='store_true',
                        help='Have a higher learning rate for final classifier')
    parser.add_argument('--num_blocks', type=int, default=12,
                        help='Number of self-attention blocks in VIT')
    parser.add_argument('--patch_size', type=int, default=4,
                        help='patch size for Transformers')

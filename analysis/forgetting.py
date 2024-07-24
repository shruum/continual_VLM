from __future__ import print_function
import os

import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.offsetbox import AnchoredText
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from cl_datasets import ContinualDataset
from cl_datasets import get_dataset
from argparse import Namespace
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# Audio
num_tasks = 10
audio_dir = r'/data/output-ai/fahad.sarfraz/mm_vggsound/results/multimodal-class-il/seq_vggsound/mm_er/er-vggsound-a-1000-param-v1-s-1'
video_dir = r'/data/output-ai/fahad.sarfraz/mm_vggsound/results/multimodal-class-il/seq_vggsound/mm_er/er-vggsound-v-1000-param-v1-s-1'
av_dir = r'/data/output-ai/fahad.sarfraz/mm_vggsound/results/multimodal-class-il/seq_vggsound/mm_er/er-vggsound-av-1000-param-v1-s-1'

audio_log = os.path.join(audio_dir, 'logs.csv')
video_log = os.path.join(video_dir, 'logs.csv')
av_log = os.path.join(av_dir, 'logs.csv')


audio_df = pd.read_csv(audio_log)
video_df = pd.read_csv(video_log)
av_df = pd.read_csv(av_log)

audio_acc = [audio_df[f'accmean_task{n + 1}'].item() for n in range(num_tasks)]
video_acc = [video_df[f'accmean_task{n + 1}'].item() for n in range(num_tasks)]
av_acc = [av_df[f'accmean_task{n + 1}'].item() for n in range(num_tasks)]

n_rows, n_cols = 1, 3
fig, ax = plt.subplots(n_rows, n_cols, figsize=(12, 4), sharey=False)

x_labels = [f"Task {i}" for i in range(1, num_tasks + 1)]

ax[0].plot(x_labels, audio_acc, color='#bc4749', label='Audio', linewidth=3)
ax[0].plot(x_labels, video_acc, color='#669bbc', label='Video', linewidth=3)
ax[0].plot(x_labels, av_acc, color='#6a994e', label='MultiModal', linewidth=3)

ax[0].set_ylabel('Accuracy (%)', fontsize=14)
ax[0].set_title('Mean Accuracy', fontsize=16)
ax[0].legend()

plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")









plt.show()


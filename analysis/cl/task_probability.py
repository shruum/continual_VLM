from __future__ import print_function
import os
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
from cl_datasets.seq_vggsound import VGGSound
from backbone.AudioVideoNet import AVClassifier


def get_task_probabilities(model, device, data_loader, task_dist):
    model.eval()
    lst_logits = []

    count = 0
    total = len(data_loader)
    with torch.no_grad():
        for (audio, video), labels, _ in data_loader:
            audio, video, labels = audio.to(device), video.to(device), labels.to(device)
            a_out, v_out, outputs = model(audio.unsqueeze(1), video)
            lst_logits.append(outputs.detach().cpu())
            count += 1

            print(f'{count} of {total} Completed')

    logits = torch.cat(lst_logits).to(device)
    softmax_scores = F.softmax(logits, dim=1)

    lst_prob = []
    for task in task_dist:
        prob = torch.mean(softmax_scores[:, task_dist[task][0]: task_dist[task][1]])
        lst_prob.append(prob.item())

    np_prob = np.array(lst_prob)
    np_prob = np_prob / np.sum(np_prob)

    return np_prob


# =============================================================================
# Load Dataset
# =============================================================================
vggsound_csv_path = r'/data/input-ai/datasets/VGGSound/data/vggsound/data_processed/vgg_seq_dataset_capped.csv'
dataset_dir = r'/data/input-ai/datasets/VGGSound/data/vggsound/data_processed'
fps = 1
num_video_frames = 4

test_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# train_dataset = VGGSound(
#     csv_path=vggsound_csv_path,
#     dataset_dir=dataset_dir,
#     mode='train',
#     fps=fps,
#     num_video_frames=num_video_frames,
#     transform=test_transform
# )

# test_dataset = VGGSound(
#     csv_path=vggsound_csv_path,
#     dataset_dir=dataset_dir,
#     mode='train',
#     fps=fps,
#     num_video_frames=num_video_frames,
#     transform=test_transform
# )
#
# data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

task_dist = {
    'task1': (0, 10),
    'task2': (10, 20),
    'task3': (20, 30),
    'task4': (30, 40),
    'task5': (40, 50),
    'task6': (50, 60),
    'task7': (60, 70),
    'task8': (70, 80),
    'task9': (80, 90),
    'task10': (90, 100),
}
# =============================================================================
# Audio Model
# =============================================================================
audio_dir = r'/data/output-ai/fahad.sarfraz/mm_vggsound/results/multimodal-class-il/seq_vggsound/mm_er/er-vggsound-a-1000-param-v1-s-1'
num_classes = 100
fusion_method = 'gated'
modalities_used = 'audio'
model_path = os.path.join(audio_dir, 'model.ph')
model_dict = torch.load(model_path)
model = AVClassifier(n_classes=num_classes, fusion=fusion_method, modalities_used=modalities_used)
model.load_state_dict(model_dict['net'])
model = model.to('cuda')
# audio_prob = get_task_probabilities(model, 'cuda', data_loader, task_dist)
audio_prob = np.array([0.01868019, 0.01725271, 0.01772954, 0.0231458, 0.03545961, 0.03243359, 0.03540539, 0.04997419, 0.06577044, 0.70414852])
# =============================================================================
# Video Model
# =============================================================================
video_dir = r'/data/output-ai/fahad.sarfraz/mm_vggsound/results/multimodal-class-il/seq_vggsound/mm_er/er-vggsound-v-1000-param-v1-s-1'
num_classes = 100
fusion_method = 'gated'
modalities_used = 'video'
model_path = os.path.join(video_dir, 'model.ph')
model_dict = torch.load(model_path)
model = AVClassifier(n_classes=num_classes, fusion=fusion_method, modalities_used=modalities_used)
model.load_state_dict(model_dict['net'])
model = model.to('cuda')
# video_prob = get_task_probabilities(model, 'cuda', data_loader, task_dist)
video_prob = np.array([0.019796, 0.01499977, 0.02457282, 0.02403626, 0.02800598, 0.02704128, 0.03405059, 0.04305073, 0.04073309, 0.74371347])

# =============================================================================
# MultiModal Model
# =============================================================================
multimodal_dir = r'/data/output-ai/fahad.sarfraz/mm_vggsound/results/multimodal-class-il/seq_vggsound/mm_er/er-vggsound-av-1000-param-v1-s-1'
num_classes = 100
fusion_method = 'film'
modalities_used = 'audio_video'
model_path = os.path.join(multimodal_dir, 'model.ph')
model_dict = torch.load(model_path)
model = AVClassifier(n_classes=num_classes, fusion=fusion_method, modalities_used=modalities_used)
model.load_state_dict(model_dict['net'])
model = model.to('cuda')

av_prob = np.array([0.03987338, 0.03585422, 0.03164141, 0.03836288, 0.05553698,
       0.05240509, 0.07481832, 0.07957544, 0.09074726, 0.50118503])
# av_prob = get_task_probabilities(model, 'cuda', data_loader, task_dist)


lst_colors = [
    '#99e2b4',
    '#88d4ab',
    '#78c6a3',
    '#67b99a',
    '#56ab91',
    '#469d89',
    '#358f80',
    '#248277',
    '#14746f',
    '#036666',

    # '#fafa6e',
    # '#c4ec74',
    # '#92dc7e',
    # '#64c987',
    # '#39b48e',
    # '#089f8f',
    # '#00898a',
    # '#08737f',
    # '#215d6e',
    # '#2a4858',
]


prob = np.vstack((audio_prob, video_prob, av_prob))
n_methods, n_tasks = prob.shape

fig, ax = plt.subplots(figsize=(7, 5))
barWidth = 0.08
for i in range(n_tasks):
    x = np.arange(n_methods) + i * barWidth
    plt.bar(x, prob[:, i], color=lst_colors[i], width=barWidth, label=f'Task {i + 1}')

plt.ylabel('Task Probability', fontsize=17)
plt.xticks([r + 5 * barWidth for r in range(n_methods)], ['Audio', 'Video', 'Multimodal'], fontsize=15)
plt.yticks(fontsize=14)
plt.legend()
plt.show()
fig.savefig(f'analysis/figures/task_prob.png', bbox_inches='tight')
fig.savefig(f'analysis/figures/task_prob.pdf', bbox_inches='tight')

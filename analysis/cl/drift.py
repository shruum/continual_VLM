import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from backbone.ResNet_mam import resnet18mam

device = 'cuda'
lst_colors = [
    # '#f9dbbd',
    "#ffffff",
    "#e1e5f2",
    '#bfdbf7',
    '#1f7a8c',
    '#022b3a',
    # '#450920',
]
from matplotlib.colors import LinearSegmentedColormap
custom1 = LinearSegmentedColormap.from_list(
    name='pink',
    colors=lst_colors,
)

dataset = 'cifar10'
lst_methods = {
    'er': '/volumes1/vlm-cl/final/results/class-il/seq-cifar10/er/model-er-l0.1-resnet18mam-seq-cifar10-buf-200-s-2/model_task%s.ph',
    'vl_er': '/volumes1/vlm-cl/final/results/class-il/seq-cifar10/vl_er/revproj-vl_er-resnet18mam-seq-cifar10-desc-200--e-50-l0.05-14.0-text-sent_transf-s-42/model_task%s.ph',
}
#
# dataset = 'cifar100'
# lst_methods = {
#     'der': '/data/output-ai/shruthi.gowda/continual/baseline/results/class-il/seq-cifar100/derpp/cll-derpp-200-seq-cifar100-s0/net_%s.pth',
#     'er': '/data/output-ai/shruthi.gowda/continual/baseline/results/class-il/seq-cifar100/er/cll-er-200-seq-cifar100-s0/net_%s.pth',
#     'cls-er':'/data/output-ai/fahad.sarfraz/baseline/task_models/seq-cifar100/c100-5-200-param-v1-s-0/task_%s_stable_model.ph',
#     # 'cls-er': '/data/output-ai/fahad.sarfraz/lll_baselines/task_models/seq-cifar100/c100-5-200-param-v4-0.05-0.1s-0/stable_model.ph',
#     'aux': '/data/output-ai/shruthi.gowda/continual/cifar100/results/class-il/seq-cifar100/derpp_mm_eman1/cll-cif100200-a0.15b0.5-lr0.03-l20.10.01-up0.06-urgent-s0_ema_net1/ema_net1_%s.pth'
# }
# dataset = 'domainnet'
# lst_methods = {
#     'der': '/data/output-ai/shruthi.gowda/continual/domainNet/base/base_tiny_params/tiny_param/results/class-il/domain-net/er/cll-er-500-domain-netv2-mod-s0/net_%s.pth',
#     'er': '/data/output-ai/shruthi.gowda/continual/domainNet/base/base_tiny_params/tiny_param/results/class-il/domain-net/derpp/cll-derpp-500-domain-netv2-s0/net_%s.pth',
#     'cls-er':'/data/output-ai/shruthi.gowda/continual/base/clser/results/class-il/domain-net/clser/clser-500-domain-net-lr0.05-0.080.05-mod-s0/net_%s.pth',
#     'aux': '/data/output-ai/shruthi.gowda/continual/domainNet/method/results/class-il/domain-net/derpp_mm_eman1/cll-500-domain-netv2-lr0.03-l20.10.01-em0.06-urgent-s0_ema_net1/ema_net1_%s.pth'
# }

method = 'aux'
buffer_size = 200
num_tasks = 5
lst_models = ['er', 'vl_er']

def get_normalized_params(model):
    lst_params = []
    for param in model.parameters():
        lst_params.append(param.view(-1) / param.max())
    return torch.cat(lst_params)

lst_diff = []
for method in lst_models:
    lst_params = []
    for task_id in range(num_tasks):
        # model_path = lst_methods[method]
        model_path = lst_methods[method] % (task_id + 1)
        model = resnet18mam(10)
        model_dict = torch.load(model_path)
        model.load_state_dict(model_dict['net'])
        model = model.to(device)        #lst_params.append(model.get_params() / model.get_params().max())
        lst_params.append(list(model.parameters()))
    sim_mat = np.zeros((num_tasks, num_tasks))
    sim = torch.nn.CosineSimilarity(dim=0, eps=1e-08)

    for i in range(num_tasks):
        for j in range(num_tasks):
            if i >= j:
                model_diff = 0
                count = 0
                for param1, param2 in zip(lst_params[i], lst_params[j]):
                    # param1, param2 = param1.view(-1), param2.view(-1)
                    param1, param2 = param1.view(-1) / param1.max(), param2.view(-1) / param2.max()
                    model_diff += sim(param1, param2)
                    # model_diff += ((param1 - param2)**2).sum()
                    # model_diff += (abs(param1 - param2)).sum()
                    count += 1
                model_diff /= count
                sim_mat[i][j] = model_diff
                print(sim_mat)

    lst_diff.append(sim_mat)

v_max = 0
v_min = 1000
for mat in lst_diff:
    mat[mat == 0] = mat.max()
    max, min = mat.max(), mat.min()
    if v_max < max:
        v_max = max
    if v_min > min:
        v_min = min

import seaborn as sns
# fig, ax = plt.subplots()

n_rows, n_cols = 1, 2
fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, 9), sharey=True, sharex=True)
annot = True
fmt = '.2f'
# fmt = '%d'
fontsize = 16
# plt.rcParams['text.usetex'] = True
x_labels = [f"T{i}" for i in range(1, num_tasks + 1)]
y_labels = [f"T{i}" for i in range(1, num_tasks + 1)]

for n, mat in enumerate(lst_diff):
    matrix = np.triu(np.ones_like(mat)) - np.identity(mat.shape[0])
    if n == 3:
        # cbar_ax = fig.add_axes([0.91, 0.18, 0.02, 0.6])
        im = sns.heatmap(mat, ax=ax[n], vmax=v_max, vmin=v_min, mask=matrix, annot=annot, cmap=custom1, alpha=0.85, cbar=False, fmt=fmt, linewidths=.5, annot_kws={"size": fontsize})
    else:
        im = sns.heatmap(mat, ax=ax[n], vmax=v_max, vmin=v_min, mask=matrix, annot=annot, cmap=custom1, alpha=0.85, cbar=False, fmt=fmt, linewidths=.5, annot_kws={"size": fontsize})
    # if n == 0:
    ax[n].set_yticks(np.arange(len(y_labels)) + 0.5)
    ax[n].set_yticklabels(y_labels, rotation=0, va='center', fontsize=fontsize)
    # else:
    #     ax[n].set_yticks([])
    ax[n].set_aspect('equal', adjustable='box')
    ax[n].set_xticks(np.arange(len(x_labels)) + 0.5)
    ax[n].set_xticklabels(x_labels, ha='center', fontsize=fontsize)

    ax[n].axhline(y=0, color='k', linewidth=1)
    ax[n].axhline(y=mat.shape[1], color='k', linewidth=1)
    ax[n].axvline(x=0, color='k', linewidth=1)
    ax[n].axvline(x=mat.shape[1], color='k', linewidth=1)

font = 21
ax[0].set_title('Replay-CL', fontsize=font)
ax[1].set_title('LG-CL', fontsize=font)

# ax = sns.heatmap(sim_mat, annot=True)
plt.subplots_adjust(wspace=0.01, hspace=0.1)
fig.tight_layout()
fig.suptitle('Representation Drift', fontsize=22)
fig.savefig(f'/volumes1/vlm-cl/paper/drift.png', bbox_inches='tight', dpi=1000)
# fig.savefig(f'/volumes2/continual_learning/paper/analysis/new/drift_dom_500.pdf', bbox_inches='tight', dpi=1000)
plt.show()

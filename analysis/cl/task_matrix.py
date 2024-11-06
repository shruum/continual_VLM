import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# lst_colors = [
#     '#f9dbbd',
#     '#ffa5ab',
#     '#da627d',
#     '#a53860',
#     '#450920',
# ]
lst_colors = [
    # '#f9dbbd',
    "#ffe5d9",
    "#ffcfd2",
    '#ffa5ab',
    '#da627d',
    '#a53860',
    # '#450920',
]
from matplotlib.colors import LinearSegmentedColormap
custom1 = LinearSegmentedColormap.from_list(
    name='pink',
    colors=lst_colors,
)

dataset = 'cifar10'

if dataset == 'cifar10':
    lst_methods = {
        'ER': '/data/output-ai/fahad.sarfraz/SYNERgy/analysis_models/cifar10/er/results/class-il/seq-cifar10/er/er-c10-b-%s-r-2',
        'DER++': '/data/output-ai/fahad.sarfraz/SYNERgy/analysis_models/cifar10/der/results/class-il/seq-cifar10/derpp/der-c10-b-%s-r-2/',
        'cls-er': '/data/output-ai/fahad.sarfraz/SYNERgy/analysis_models/cifar10/cil_er/results/class-il/seq-cifar10/dualmeanerv6/er-c10-%s-param-v1-s-0_stable_ema_model/',
        'aux': '/data/output-ai/shruthi.gowda/continual/method/clmm_derpp_v2n1/results/class-il/seq-cifar10/derpp_mm_eman1_model/derppema-n1m-%s-seq-cifar10-kl1.0-0.1-up0.2-s2_ema_net1/',
        'aux_w': '/data/output-ai/shruthi.gowda/continual/method/clmm_derpp_v2n1/results/class-il/seq-cifar10/derpp_mm_eman1_model/derppema-n1m-%s-seq-cifar10-kl1.0-0.1-up0.2-s2/',
        'shape': '/data/output-ai/shruthi.gowda/continual/method/clmm_derpp_v2n1/results/class-il/seq-cifar10/derpp_mm_eman1_model/derppema-n1m-%s-seq-cifar10-kl1.0-0.1-up0.2-s2_net2/'
        # 'aux': '/data/output-ai/shruthi.gowda/continual/clmm_derpp_v2n1_l2/results/class-il/seq-cifar10/derpp_mm_eman1/derppema-n1-%s-seq-cifar10-l20.10.1-up0.2-s0_ema_net1/',
        # 'aux_w': '/data/output-ai/shruthi.gowda/continual/clmm_derpp_v2n1_l2/results/class-il/seq-cifar10/derpp_mm_eman1/derppema-n1-%s-seq-cifar10-l20.10.1-up0.2-s0/',
        # 'shape': '/data/output-ai/shruthi.gowda/continual/clmm_derpp_v2n1_l2/results/class-il/seq-cifar10/derpp_mm_eman1/derppema-n1-%s-seq-cifar10-l20.10.1-up0.2-s0_net2/',

    }
    num_tasks = 5
    annot = True
elif dataset == 'cifar100':
    lst_methods = {
        'DER++': '/data/output-ai/fahad.sarfraz/lll_baselines/results/class-il/seq-cifar100/derpp/derpp-c100-%s-0.03-0.1-0.1-s0/', #derpp-c100-%s-0.03-0.1-0.2-s0-analysis/',
        'ER': '/data/output-ai/fahad.sarfraz/lll_baselines/results/class-il/seq-cifar100/er/er-c100-%s-0.1-s0/',
        'cls-er': '/data/output-ai/fahad.sarfraz/lll_baselines/results/class-il/seq-cifar100/clser/c100-5-%s-param-v3-0.05-0.1s-1_stable_model/', #c100-5-%s-param-v4-0.05-0.1s-1_stable_model/'
        'aux': '/data/output-ai/shruthi.gowda/continual/cifar100/results/class-il/seq-cifar100/derpp_mm_eman1/cll-cif100%s-a0.15b0.5-lr0.03-l20.10.01-up0.09-urgent-s1_ema_net1/',
        'aux_w': '/data/output-ai/shruthi.gowda/continual/cifar100/results/class-il/seq-cifar100/derpp_mm_eman1/cll-cif100%s-a0.15b0.5-lr0.03-l20.10.01-up0.09-urgent-s1/',
        'shape': '/data/output-ai/shruthi.gowda/continual/cifar100/results/class-il/seq-cifar100/derpp_mm_eman1/cll-cif100%s-a0.15b0.5-lr0.03-l20.10.01-up0.09-urgent-s1_net2'
    }
    num_tasks = 5
    annot = True
elif dataset == 'domainnet':
    lst_methods = {
        'DER++': '/data/output-ai/shruthi.gowda/continual/domain_net/base/tiny_param/results/domain-il/domain-net/derpp/cll-derpp-%s-domain-netv2-s0',
        'ER': '/data/output-ai/shruthi.gowda/continual/domain_net/base/tiny_param/results/domain-il/domain-net/er/cll-er-%s-domain-netv2-s0',
        'cls-er': '/data/output-ai/shruthi.gowda/continual/domain_net/results/domain-il/domain-net/clser/cll-cls-%s-domain-netv2-64-l20.1-0.01-s0_stable_model',
        'aux': '/data/output-ai/shruthi.gowda/continual/domain_net/results/domain-il/domain-net/derpp_mm_eman1/cll-%s-domain-netv2-lr0.03-l20.10.01-em0.06-s2_ema_net1',
        'aux_w': '/data/output-ai/shruthi.gowda/continual/domain_net/results/domain-il/domain-net/derpp_mm_eman1/cll-%s-domain-netv2-lr0.03-l20.10.01-em0.06-s2',
        'shape': '/data/output-ai/shruthi.gowda/continual/domain_net/results/domain-il/domain-net/derpp_mm_eman1/cll-%s-domain-netv2-lr0.03-l20.10.01-em0.06-s2_net2'
    }
    num_tasks = 6
    annot = True

x_labels = [f"T{i}" for i in range(1, num_tasks + 1)]
y_labels = [f"After T{i}" for i in range(1, num_tasks + 1)]

n_rows, n_cols = 2, 3
fig, ax = plt.subplots(n_rows, n_cols, figsize=(14, 9), sharey=True, sharex=True)

annot = True
fmt = '.1f'
# fmt = '%d'
font = 14

lst_method = ['ER', 'DER++', 'cls-er', 'aux', 'aux_w', 'shape']
buffer_size = 200

# Get Max and Min
v_max = 0
v_min = 1000
for n, method in enumerate(lst_method):
    perf_path = os.path.join(lst_methods[method] % buffer_size, 'task_performance.txt')

    np_perf = np.loadtxt(perf_path)
    matrix = np.triu(np.ones_like(np_perf)) - np.identity(np_perf.shape[0])
    max, min = np_perf.max(), np_perf.min()

    if v_max < max:
        v_max = max
    if v_min > min:
        v_min = min


x_labels = [f"T{i}" for i in range(1, num_tasks + 1)]
y_labels = [f"After T{i}" for i in range(1, num_tasks + 1)]

k = 0
for n, method in enumerate(lst_method):
    perf_path = os.path.join(lst_methods[method] % buffer_size, 'task_performance.txt')

    np_perf = np.loadtxt(perf_path)

    if n > 2:
        k = 1
        n -= 3
    if n == 2:
        #cbar_ax = fig.add_axes([0.91, 0.18, 0.02, 0.6])
        im = sns.heatmap(np_perf, ax=ax[k][n], vmax=v_max, vmin=v_min, mask=matrix, annot=annot, cmap=custom1, alpha=0.85, cbar=False, fmt=fmt, linewidths=.5, annot_kws={"size": font}) #, cbar_ax=cbar_ax)
    else:
        im = sns.heatmap(np_perf, ax=ax[k][n], vmax=v_max, vmin=v_min, mask=matrix, annot=annot, cmap=custom1, alpha=0.85, cbar=False, fmt=fmt, linewidths=.5, annot_kws={"size": font})
    ax[k][n].set_xticks(np.arange(len(x_labels)) + 0.5)
    ax[k][n].set_yticks(np.arange(len(y_labels)) + 0.5)
    ax[k][n].set_xticklabels(x_labels, ha='center', fontsize=font)
    ax[k][n].set_yticklabels(y_labels, rotation=0, va='center', fontsize=font)
    ax[k][n].set_aspect('equal', adjustable='box')

    ax[k][n].axhline(y=0, color='k', linewidth=1)
    ax[k][n].axhline(y=np_perf.shape[1], color='k', linewidth=2)
    ax[k][n].axvline(x=0, color='k', linewidth=1)
    ax[k][n].axvline(x=np_perf.shape[1], color='k', linewidth=2)



ax[0][0].set_title('ER', fontsize=font+3)
ax[0][1].set_title('DER++', fontsize=font+3)
ax[0][2].set_title('CLS-ER', fontsize=font+3)
ax[1][0].set_title('DUCA (SM - Default)', fontsize=font)
ax[1][1].set_title('DUCA (WM)', fontsize=font)
ax[1][2].set_title('DUCA (IBL)', fontsize=font)
# ax[1][0].set_position([0.24,0.125,0.228,0.343])
# ax[1][1].set_position([0.55,0.125,0.228,0.343])


# fig.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.25)
plt.show()

fig.savefig(f'/volumes2/continual_learning/paper/icml/task_all_cif10.pdf', bbox_inches='tight')
# fig.savefig(f'/volumes2/continual_learning/paper/analysis/new/task_all_dom_500.pdf', dpi=600, bbox_inches='tight')

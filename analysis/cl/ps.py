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
    "#bfdbf7",
    "#1f7a8c",
    '#ffb703',
]
from matplotlib.colors import LinearSegmentedColormap
custom1 = LinearSegmentedColormap.from_list(
    name='pink',
    colors=lst_colors,
)

dataset = 'domainnet'

if dataset == 'cifar10':
    lst_methods = {
        'Replay-CL': r'/volumes1/vlm-cl/baseline_final/results/class-il/seq-cifar10/er/er-resnet50mam-seq-cifar10-buf-200-s-42/task_performance.txt',
        'LG-Ex': r'/volumes1/vlm-cl/results_final/results/class-il/seq-cifar10/vl_er/revproj-vl_er-resnet18mam-seq-cifar10-desc-200--lr-0.03-l-sim-12.0-text-sent_transf-s-42/task_performance.txt',
        'LG-Ix': r'/volumes1/vlm-cl/fahad/all_results_29_09/results_implicit_cuda1/class-il/seq-cifar10/er/lgix-er-resnet18mamllm-sent_transf-seq-cifar10-10-b-200--l0.005-wd0.01-e-100-l0.005--s-42/task_performance.txt',
    }
    num_tasks = 5
    annot = True
elif dataset == 'cifar100':
    lst_methods = {
        'DER++': '/data/output-ai/shruthi.gowda/continual/baseline/results/class-il/seq-cifar100/derpp/cll-derpp-%s-seq-cifar100-s0/',
        'ER': '/data/output-ai/shruthi.gowda/continual/baseline/results/class-il/seq-cifar100/er/cll-er-%s-seq-cifar100-s0/',
        'cls-er': '/data/output-ai/fahad.sarfraz/lll_baselines/results/class-il/seq-cifar100/clser/c100-5-%s-param-v4-0.05-0.1s-0_stable_model/',
        'aux': '/data/output-ai/shruthi.gowda/continual/cifar100/results/class-il/seq-cifar100/derpp_mm_eman1/cll-cif100%s-a0.1b0.5-lr0.03-l20.10.01-up0.06-g4-s0_ema_net1/',
    }
    num_tasks = 5
    annot = True
elif dataset == 'domainnet':
    lst_methods = {
        'Replay-CL': r'/volumes1/vlm-cl/baseline/results/domain-il/dn4il/er/er-dn4il-buf-200-s-2/task_performance.txt',
        'LG-Ex': r'/volumes1/vlm-cl/snel/results_final/results/domain-il/dn4il/vl_er/aft-vl_er-dn4il-b200-lr-0.03-e100-8.0-t-sent_transf-s-1/task_performance.txt',
        'LG-Ix': r'/volumes1/vlm-cl/snel/results_final/results/domain-il/dn4il/er/ix-normal-resnet18mamllm-dn4il-b500-lr-0.003-wd-0.01-s-1/task_performance.txt',
    }
    num_tasks = 6
    annot = True

x_labels = [f"T{i}" for i in range(1, num_tasks + 1)]
y_labels = [f"After T{i}" for i in range(1, num_tasks + 1)]
n_rows, n_cols = 1, 1
fig, ax = plt.subplots(n_rows, n_cols, figsize=(13, 11)) # sharey=True, sharex=True)
annot = True
fmt = '.1f'
# fmt = '%d'
font = 18

lst_method = ['Replay-CL', 'LG-Ex', 'LG-Ix']
buffer_size = 200
k = 0
x =  np.arange(3)
pl = []
st = []
tr = []
for n, method in enumerate(lst_method):
    perf_path = lst_methods[method]
    np_perf = np.loadtxt(perf_path)
    p = np_perf[0][0] + np_perf[1][1] + np_perf[2][2] + np_perf[3][3] + np_perf[4][4]
    pl.append(p/5)
    s = np_perf[4][0] + np_perf[4][1] + np_perf[4][2] + np_perf[4][3]
    st.append(s/4)
    t =(1/p)+(1/s)
    tr.append(1/t)

width = 0.4
ax.bar(x - width/2, pl, width, color='#b56576', align='center', label='Plasticity', alpha=.99)
ax.bar(x + width/2, st, width, color='#6d597a', align='center', label='Stability', alpha=.99)
# ax.bar(x + 0.2, tr, width=0.2, color='#a53860', align='center', label='Trade-off')
import matplotlib as mpl
mpl.rcParams["hatch.color"] = 'red'
font = 34
y_ticks = ax.yaxis.get_major_ticks()
plt.ylim([0, 100])
# plt.yticks(np.arange(0, 90, 10))
# plt.yticks[2].set_visible(False)
# y_ticks[-1].label1.set_visible(False)
plt.xticks(x, ['Base', 'ExLG', 'IxLG'])
plt.xticks(fontsize=35)
plt.ylabel("Accuracy", fontsize=font)
plt.legend(fontsize=36, loc="upper center", ncol=2, frameon=False, )
plt.tick_params(axis="y", labelsize=font)      # To change the y-axis
plt.show()

# fig.savefig(f'/volumes1/vlm-cl/paper/ps_cif10.png', bbox_inches='tight')

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


lst_methods = {
    'Replay-CL': r'/volumes1/vlm-cl/baseline_final/results/class-il/seq-cifar10/er/er-l0.1-resnet50mam-seq-cifar10-buf-200-s-42/task_performance.txt',
    'LG-ex': r'/volumes1/vlm-cl/snel/results_final/results/class-il/seq-cifar10/vl_er/vl_er-seq-cifar10-b200-lr-0.05-l-before-50.0-sent_transf-s-0/task_performance.txt',
    'LG-ix': r'/volumes1/vlm-cl/fahad/all_results_29_09/results_implicit_cuda1/class-il/seq-cifar10/er/lgix-er-resnet18mamllm-sent_transf-seq-cifar10-5-b-200--l0.001-wd0.01-e-100-l0.001--s-42/task_performance.txt',
    # 'LG-ix' : r'/volumes1/vlm-cl/fahad/all_results_29_09/results_implicit_cuda1/class-il/seq-cifar10/er/lgix-er-resnet18mamllm-sent_transf-seq-cifar10-5-b-200--l0.005-wd0.01-e-100-l0.005--s-42/task_performance.txt'
}

num_tasks = 5
annot = True
x_labels = ['T1', 'T2', 'T3', 'T4', 'T5']
y_labels = [f"After {i}" for i in x_labels]

fmt = '.1f'
# fmt = '%d'
fontsize =11

lst_colors = [
    # '#f9dbbd',
    # "#ffffff",
    "#e1e5f2",
    '#bfdbf7',
    "#4ea8de",
    '#219ebc',
    '#022b3a',
    # '#450920',
]

# lst_colors = ['#ADD7F6', '#87BFFF', '#3F8EFC', '#2667FF', '#3B28CC', '#022b3a']
from matplotlib.colors import LinearSegmentedColormap
custom1 = LinearSegmentedColormap.from_list(
    name='pink',
    colors=lst_colors,
)

lst_method = ['Replay-CL', 'LG-ex', 'LG-ix']

n_rows, n_cols = 1, 3
fig, ax = plt.subplots(n_rows, n_cols, figsize=(13, 5), sharey=True, sharex=True)
# Get Max and Min
v_max = 0
v_min = 1000
for n, method in enumerate(lst_method):
    perf_path = lst_methods[method]
    np_perf = np.loadtxt(perf_path)
    max, min = np_perf.max(), np_perf.min()
    if v_max < max:
        v_max = max
    if v_min > min:
        v_min = min

for n, method in enumerate(lst_method):
    perf_path = lst_methods[method]
    np_perf = np.loadtxt(perf_path)
    mask = np.triu(np.ones_like(np_perf)) - np.identity(np_perf.shape[0])
    if n == 3:
        im = sns.heatmap(np_perf, ax=ax[n], vmax=v_max, vmin=v_min, mask=mask, annot=annot, cmap=custom1, alpha=0.85, cbar=False, fmt=fmt, linewidths=.5, annot_kws={"size": fontsize})
    else:
        im = sns.heatmap(np_perf, ax=ax[n], vmax=v_max, vmin=v_min, mask=mask, annot=annot, cmap=custom1, alpha=0.85, cbar=False, fmt=fmt, linewidths=.5, annot_kws={"size": fontsize})

    ax[n].set_xticks(np.arange(len(x_labels)) + 0.5)
    ax[n].set_yticks(np.arange(len(y_labels)) + 0.5)
    ax[n].set_xticklabels(x_labels, ha='center', fontsize=12)
    ax[n].set_yticklabels(y_labels, rotation=0, va='center', fontsize=12)
    ax[n].set_aspect('equal', adjustable='box')

    ax[n].axhline(y=0, color='k', linewidth=1)
    ax[n].axhline(y=np_perf.shape[1], color='k', linewidth=2)
    ax[n].axvline(x=0, color='k', linewidth=1)
    ax[n].axvline(x=np_perf.shape[1], color='k', linewidth=2)

ax[0].set_title('ER', fontsize=16)
ax[1].set_title('ExLG', fontsize=16)
ax[2].set_title('ImLG', fontsize=16)

plt.subplots_adjust(wspace=0.001, hspace=0.01)
fig.tight_layout()
plt.show()

fig.savefig(f'/volumes1/vlm-cl/paper/task_perf_cif10.png', bbox_inches='tight')

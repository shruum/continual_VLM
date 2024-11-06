import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


lst_methods = {
    'Audio': r'/data/output-ai/fahad.sarfraz/mm_vggsound/results/multimodal-class-il/seq_vggsound/mm_er/er-vggsound-a-1000-param-v1-s-1/task_performance.txt',
    'Video': r'/data/output-ai/fahad.sarfraz/mm_vggsound/results/multimodal-class-il/seq_vggsound/mm_er/er-vggsound-v-1000-param-v1-s-1/task_performance.txt',
    'Multimodal': r'/data/output-ai/fahad.sarfraz/mm_vggsound/results/multimodal-class-il/seq_vggsound/mm_er/er-vggsound-av-1000-param-v1-s-1/task_performance.txt',
}

num_tasks = 10
annot = True

x_labels = [f"T{i}" for i in range(1, num_tasks + 1)]
y_labels = [f"After T{i}" for i in range(1, num_tasks + 1)]

fmt = '.1f'
# fmt = '%d'
fontsize =11

lst_method = ['Audio', 'Video', 'Multimodal']

n_rows, n_cols = 1, 3
fig, ax = plt.subplots(n_rows, n_cols, figsize=(14, 5), sharey=True, sharex=True)
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

x_labels = [f"T{i}" for i in range(1, num_tasks + 1)]
y_labels = [f"After T{i}" for i in range(1, num_tasks + 1)]

for n, method in enumerate(lst_method):
    perf_path = lst_methods[method]

    np_perf = np.loadtxt(perf_path)

    mask = np.triu(np.ones_like(np_perf)) - np.identity(np_perf.shape[0])
    if n == 3:
        im = sns.heatmap(np_perf, ax=ax[n], vmax=v_max, vmin=v_min, mask=mask, annot=annot, cmap=sns.color_palette("crest", as_cmap=True), alpha=0.85, cbar=False, fmt=fmt, linewidths=.5, annot_kws={"size": fontsize})
    else:
        im = sns.heatmap(np_perf, ax=ax[n], vmax=v_max, vmin=v_min, mask=mask, annot=annot, cmap=sns.color_palette("crest", as_cmap=True), alpha=0.85, cbar=False, fmt=fmt, linewidths=.5, annot_kws={"size": fontsize})

    ax[n].set_xticks(np.arange(len(x_labels)) + 0.5)
    ax[n].set_yticks(np.arange(len(y_labels)) + 0.5)
    ax[n].set_xticklabels(x_labels, ha='center', fontsize=12)
    ax[n].set_yticklabels(y_labels, rotation=0, va='center', fontsize=12)
    ax[n].set_aspect('equal', adjustable='box')

    ax[n].axhline(y=0, color='k', linewidth=1)
    ax[n].axhline(y=np_perf.shape[1], color='k', linewidth=2)
    ax[n].axvline(x=0, color='k', linewidth=1)
    ax[n].axvline(x=np_perf.shape[1], color='k', linewidth=2)

ax[0].set_title('Audio', fontsize=16)
ax[1].set_title('Visual', fontsize=16)
ax[2].set_title('Multimodal', fontsize=16)


plt.subplots_adjust(wspace=0.001, hspace=0.01)
fig.tight_layout()
plt.show()

fig.savefig(f'analysis/figures/task_performance.png', bbox_inches='tight')
fig.savefig(f'analysis/figures/task_performance.pdf', bbox_inches='tight')
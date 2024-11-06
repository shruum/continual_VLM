import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


lst_methods = {
    'Replay-CL': r'/volumes1/vlm-cl/baseline/results/domain-il/dn4il/er/er-dn4il-buf-200-s-2/task_performance.txt',
    # 'LG-CL': r'/volumes1/vlm-cl/snel/results_final/results/domain-il/dn4il/vl_er/aft-vl_er-dn4il-b500-lr-0.05-l-sim-24.0-t-sent_transf-s-1/task_performance.txt',
    'LG-CL': r'/volumes1/vlm-cl/snel/results_final/results/domain-il/dn4il/vl_er/aft-vl_er-dn4il-b200-lr-0.03-e100-8.0-t-sent_transf-s-1/task_performance.txt',
}

num_tasks = 6
annot = True
x_labels = ['Real', 'CLip', 'Igraph', 'Paint', 'Sketch', 'Qdraw']
y_labels = [f"After {i}" for i in x_labels]

fmt = '.1f'
# fmt = '%d'
fontsize =11

lst_colors = [
    # '#f9dbbd',
    "#ffffff",
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

lst_method = ['Replay-CL', 'LG-CL']

n_rows, n_cols = 1, 2
fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 5), sharey=True, sharex=True)
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

ax[0].set_title('Base-CL', fontsize=16)
ax[1].set_title('ExLG-CL', fontsize=16)

plt.subplots_adjust(wspace=0.001, hspace=0.01)
fig.tight_layout()
plt.show()

fig.savefig(f'/volumes1/vlm-cl/paper/task_perf_domain.png', bbox_inches='tight')
fig.savefig(f'/volumes1/vlm-cl/paper/task_perf_domain.pdf', bbox_inches='tight')
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
# libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
fontsize = 10

lst_method = ['Audio', 'Video', 'Multimodal']

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

x_labels = [f"T{i}" for i in range(1, num_tasks + 1)]
y_labels = [f"After T{i}" for i in range(1, num_tasks + 1)]

results = {}

for n, method in enumerate(lst_method):
    perf_path = lst_methods[method]

    np_perf = np.loadtxt(perf_path)

    plasticity = np.mean(np.diag(np_perf))
    stability = np.mean(np_perf[-1][:-1])
    tradeoff = (2 * stability * plasticity) / (stability + plasticity)

    results[method] = [plasticity, stability, tradeoff]

plt.figure(figsize=(4, 3))
N = len(results)
ind = np.arange(N)

# set width of bars
barWidth = 0.20

plt.bar(ind, results['Audio'], barWidth, label='Audio', color='#bc4749')
plt.bar(ind + barWidth, results['Video'], barWidth, label='Video', color='#669bbc')
plt.bar(ind + 2 * barWidth, results['Multimodal'], barWidth, label='Multimodal', color='#6a994e')

plt.xticks(ind + barWidth, ('Plasticity', 'Stability', 'Tradeoff'))
plt.legend()

plt.savefig('analysis/figures/plasticity_stability.pdf', bbox_inches='tight', dpi=350)
plt.show()






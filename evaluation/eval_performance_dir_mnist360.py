import os
import glob
import pandas as pd

exp_dir = r'/data/output/fahad.sarfraz/cil_er_mnist360_final'

lst_dict_vals = []
lst_files = glob.glob(r'%s/results/*/*/*/*/mean_accs.csv' % exp_dir)

for file_path in lst_files:
    if 'plastic_ema' in file_path:
        eval_mode = 'plastic_ema'
    elif 'stable_ema' in file_path:
        eval_mode = 'stable_ema'
    elif '_ema' in file_path:
        eval_mode = 'ema'
    elif '_ensemble' in file_path:
        eval_mode = 'ensemble'
    else:
        eval_mode = 'normal'

    path_tokens = file_path.split('/')
    dataset = path_tokens[-4]
    method = path_tokens[-3]

    try:
        raw_dict_vals = pd.read_csv(file_path).to_dict()

        dict_vals = {}
        for key in raw_dict_vals.keys():
            if not '.' in key:
                dict_vals[key] = raw_dict_vals[key][0]


        dict_vals['run'] = dict_vals['experiment_id'].split('-')[-1]
        dict_vals['dataset'] = dataset
        dict_vals['eval_mode'] = eval_mode
        dict_vals['method'] = method

        if dataset in ['seq-cifar10', 'seq-mnist']:
            dict_vals['acc'] = dict_vals['task5']
        elif dataset in ['seq-cifar100', 'seq-tinyimg']:
            dict_vals['acc'] = dict_vals['task10']
        elif dataset in ['perm-mnist', 'rot-mnist', 'gil-cifar100']:
            dict_vals['acc'] = dict_vals['task20']

        lst_dict_vals.append(dict_vals)
    except Exception as e:
        print(file_path)

df = pd.DataFrame(lst_dict_vals)
df.to_csv(os.path.join(exp_dir, 'experimental_results.csv'), index=False)

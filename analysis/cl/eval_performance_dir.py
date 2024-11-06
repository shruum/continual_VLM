import os
import glob
import pandas as pd

exp_dir = r'/data/output-ai/shruthi.gowda/continual/ccl_ema_aug'
# exp_dir = r'/data/output-ai/shruthi.gowda/continual/derpp_base'

lst_tasks = ['domain-il',] #'class-il']# , 'task-il', 'domain-il']
lst_dict_vals = []
for task in lst_tasks:
    lst_files = glob.glob(r'%s/results/%s/*/*/*/mean_accs.csv' % (exp_dir, task))
    for file_path in lst_files:
        if 'ema_net1' in file_path:
            eval_mode = 'ema_net1'
        elif 'ema_net2' in file_path:
            eval_mode = 'ema_net2'
        elif 'net2' in file_path:
            eval_mode = 'net2'
        elif 'ema_lin' in file_path:
            eval_mode = 'ema_lin'
        elif 'stable_model' in file_path:
            eval_mode = 'stable_model'
        elif 'plastic_model' in file_path:
            eval_mode = 'plastic_model'
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
            dict_vals['task'] = task
            dict_vals['dataset'] = dataset
            dict_vals['eval_mode'] = eval_mode
            dict_vals['method'] = method
            if dataset in ['seq-cifar10', 'seq-cifar100', 'seq-mnist']:
                dict_vals['acc'] = dict_vals['task5']
            elif dataset in ['seq-tinyimg', 'seq-gcifar100']:
                dict_vals['acc'] = dict_vals['task10']
            elif dataset in ['perm-mnist', 'rot-mnist', 'gil-cifar100', 'gcil-gcifar100']:
                dict_vals['acc'] = dict_vals['task20']
            elif dataset in ['perm-mnist-n', 'rot-mnist']:
                dict_vals['acc'] = dict_vals['task20'] #dict_vals[f'task{dict_vals["num_tasks"]}']
            #remove for task incremental
            # if dict_vals['task'] in ['domain-il', 'class-il']:
            lst_dict_vals.append(dict_vals)
            dict_vals.pop('return_index')
            dict_vals.pop('save_model')
        except Exception as e:
            print(file_path)
df = pd.DataFrame(lst_dict_vals)
df.to_csv(os.path.join(exp_dir, 'ccl_ibl_domain.csv'), index=False)




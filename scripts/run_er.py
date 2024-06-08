import yaml
import os

job = yaml.load(open(r'/volumes1/xai_cl/multimodal_cl/nie_continual_learning/mammoth/scripts/template.yaml'), Loader=yaml.Loader)
# job = yaml.load(open(r'scripts/template_gpu4.yaml'))

best_params = {
    100: {
        'idt': 'v1',
        'lr': 0.1,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
        'loss_wt_lst1': [0.1, 0.5, 1.0, 2.0],
        'alpha': 0.1,
    },
    200: {
        'idt': 'v1',
        'lr': 0.1,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
        'loss_wt_lst1': [0.1, 0.5, 1.0, 2.0],
        'alpha': 0.1,
    },
    500: {
        'idt': 'v1',
        'lr': 0.1,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
        'loss_wt_lst1': [0.1, 0.5, 1.0, 2.0],
        'alpha': 0.1,
    },
}


lst_buffer_size = [100] #[100, 200, 500]
num_runs = 1
start_seed = 0
count = 0
dataset = "seq-cifar10"
loss_types = ['l2'] # 'kl']

for seed in range(start_seed, start_seed + num_runs):
    for buffer_size in lst_buffer_size:
        train_params = best_params[buffer_size]
        for loss_type in loss_types:
            for loss_wt in train_params['loss_wt_lst1']:
                exp_id = f"vl-er-{buffer_size}-loss-{loss_type}{loss_wt}-gpt1-s-{seed}"
                job_args = ["-c", f"python /git/mammoth/main.py  \
                    --experiment_id {exp_id} \
                    --model vl_er \
                    --dataset {dataset} \
                    --buffer_size {buffer_size} \
                    --lr {train_params['lr']} \
                    --n_epochs {train_params['n_epochs']} \
                    --minibatch_size {train_params['minibatch_size']} \
                    --batch_size {train_params['batch_size']} \
                    --output_dir /output/vision_lang \
                    --tensorboard 1 \
                    --nowand 1 \
                    --loss_type {loss_type} \
                    --loss_wt {loss_wt} {loss_wt} {loss_wt} {loss_wt} \
                    "]
                # set job params
                job['metadata']['name'] = 'shru-' + exp_id
                job['spec']['template']['spec']['containers'][0]['args'] = job_args

                yaml_out = '/volumes1/xai_cl/multimodal_cl/nie_continual_learning/mammoth/scripts/temp/%s.yaml' % exp_id

                with open(yaml_out, 'w') as outfile:
                    yaml.dump(job, outfile, default_flow_style=False)

                count += 1
                os.system('kubectl -n cyber-security-gpu l -f %s' % yaml_out)

print('%s jobs counted' % count)

# --alpha {train_params['alpha']} \

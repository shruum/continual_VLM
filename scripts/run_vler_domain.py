import subprocess
import os

# Parameters
lst_buffer_size = [200]  # [500]
num_runs = 1
start_seed = 0
dataset = "dn4il"
loss_types = ['sim', 'kl'] #, 'l2']  # ['nce', 'l2']
loss_wt_lst = [2.0, 6.0, 12.0, 15.0]
text_enc_lst = ['sent_transf'] #, 'bert', 'clip']
log_file = "error_log.txt"
lr_lst = [0.1, 0.03]

# Function to handle errors
def handle_error(exp_id):
    with open(log_file, "a") as f:
        f.write(f"Error occurred in experiment {exp_id}\n")

# Set CUDA_VISIBLE_DEVICES to use GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Iterate over parameters
for buffer_size in lst_buffer_size:
    for lr in lr_lst:
        for text_enc in text_enc_lst:
            for loss_mode in loss_types:
                for loss_wt in loss_wt_lst:
                    for seed in range(start_seed, start_seed + num_runs):
                        exp_id = f"domain-vlm-r50-lr{lr}-{buffer_size}-loss-{loss_mode}-{loss_wt}-text-{text_enc}-s-{seed}"
                        print(f"Running experiment {exp_id}")

                        command = [
                            'python', '/volumes1/vlm-cl/continual_VLM/main.py',
                            '--experiment_id', exp_id,
                            '--model', 'vl_er',
                            '--dataset', dataset,
                            '--dataset_dir', '/volumes1/datasets/DN4IL',
                            '--buffer_size', str(buffer_size),
                            '--lr', str(lr),
                            '--n_epochs', '50',
                            '--minibatch_size', '32',
                            '--batch_size', '32',
                            '--tensorboard', '1',
                            '--nowand', '1',
                            '--text_model', text_enc,
                            '--loss_wt', str(loss_wt), str(loss_wt), str(loss_wt), str(loss_wt),
                            '--ignore_other_metrics', '1',
                            '--wandb_project', 'continual_VLM',
                            '--wandb_entity', 'sngowda42',
                            '--output_dir', '/volumes1/vlm-cl/results',
                            '--loss_mode', loss_mode,
                            '--gpt_path', '/volumes1/datasets/domainnet_description_100.json'
                        ]

                        try:
                            subprocess.run(command, check=True)
                        except subprocess.CalledProcessError:
                            handle_error(exp_id)

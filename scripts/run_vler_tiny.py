import subprocess
import os
import itertools

# Define parameters
lst_buffer_size = [200]  # [100, 200, 500]
lst_lr = [0.1, 0.03]
num_runs = 1
start_seed = 0
dataset = "seq-tinyimg"
loss_types = ['sim', 'kl']
loss_wt_lst = [1.0, 0.5, 2.0, 6.0]
text_enc_lst = ['sent_transf', 'bert'] #, 'clip']
log_file = "error_log.txt"

# Create a list of all combinations
combinations = list(itertools.product(
    lst_buffer_size,
    lst_lr,
    text_enc_lst,
    loss_types,
    loss_wt_lst,
    range(start_seed, start_seed + num_runs)
))

# Function to handle errors
def handle_error(exp_id):
    print(f"Error occurred in experiment {exp_id}")
    with open(log_file, "a") as f:
        f.write(f"{exp_id}\n")

# Iterate over combinations
for buffer_size, lr, text_enc, loss_mode, loss_wt, seed in combinations:
    exp_id = (f"desk-res50-{dataset}-{buffer_size}--lr-{lr}-l-{loss_mode}-{loss_wt}-text-{text_enc}-s-{seed}")
    print(f"Running experiment {exp_id}")
    
    # Construct the command
    cmd = [
        "python", "/volumes1/vlm-cl/continual_VLM/main.py",
        "--experiment_id", exp_id,
        "--model", "vl_er",
        "--dataset", dataset,
        "--dataset_dir", "/volumes1/datasets/tiny-imagenet-200",
        "--buffer_size", str(buffer_size),
        "--lr", str(lr),
        "--n_epochs", "50",
        "--minibatch_size", "32",
        "--batch_size", "32",
        "--tensorboard", "1",
        "--nowand", "1",
        "--text_model", text_enc,
        "--loss_wt", str(loss_wt), str(loss_wt), str(loss_wt), str(loss_wt),
        "--ignore_other_metrics", "1",
        "--wandb_project", "continual_VLM",
        "--wandb_entity", "sngowda42",
        "--output_dir", "experiments_res50",
        "--loss_mode", loss_mode,
        '--gpt_path', '/volumes1/norm_datasets/tinyimagenet_description.json'
    ]
    
    try:
        # Run the command
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # Handle error
        handle_error(exp_id)

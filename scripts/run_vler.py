import subprocess
import os
import itertools

# Define parameters
lst_buffer_size = [200]  # [100, 200, 500]
lst_lr = [0.03]
num_runs = 1
start_seed = 0
dataset = "seq-cifar10"
loss_types = ['sim'] #'kl']
loss_wt_lst = [2.0, 6.0, 12.0]
text_enc_lst = ['sent_transf'] #, 'bert']
gpt_path_lst = ['cl_datasets/metadata/lvis_gpt3_text-davinci-002_descriptions_author.json', 'cl_datasets/metadata/cifar10_descriptions.json']
log_file = "error_log.txt"

# Create a list of all combinations
combinations = list(itertools.product(
    lst_buffer_size,
    lst_lr,
    text_enc_lst,
    loss_types,
    loss_wt_lst,
    gpt_path_lst,
    range(start_seed, start_seed + num_runs)
))

# Function to handle errors
def handle_error(exp_id):
    print(f"Error occurred in experiment {exp_id}")
    with open(log_file, "a") as f:
        f.write(f"{exp_id}\n")

# Iterate over combinations
for buffer_size, lr, text_enc, loss_mode, loss_wt, gpt_path, seed in combinations:

    if 'lvis_gpt3' in gpt_path:
        gpt = 'old'
    else:
        gpt = 'new'

    exp_id = (f"res50-{dataset}-desc-{gpt}-{buffer_size}--lr-{lr}-l-{loss_mode}-{loss_wt}-text-{text_enc}-s-{seed}")
    print(f"Running experiment {exp_id}")
    # Construct the command
    cmd = [
        "python", "/volumes1/vlm-cl/continual_VLM/main.py",
        "--experiment_id", exp_id,
        "--model", "vl_er",
        "--dataset", dataset,
        "--dataset_dir", "/volumes1/datasets/cifar",
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
        "--output_dir", "/volumes1/vlm-cl/results",
        "--loss_mode", loss_mode,
        '--gpt_path', gpt_path
    ]

    try:
        # Run the command
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # Handle error
        handle_error(exp_id)


All accuracies: [84.80000390625, 76.9000029296875, 70.80000189208984, 64.3250015258789, 61.24000170898437, 60.41666839599609, 56.042858372279575, 54.075001373291016, 52.85555692545573, 49.09000144042969]
Average Incremental Accuracy: 63.05450984703427

All accuracies: [84.100001953125, 77.20000219726562, 70.36666918945312, 66.02500213623047, 64.44000205078125, 64.80000231933593, 62.18571602957589, 61.087501525878906, 58.771111252848307, 55.39000119628906]
Average Incremental Accuracy: 66.43060111264184

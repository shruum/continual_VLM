import subprocess
import os
import itertools


datasets = ['cifar10_imb'] # 'cifar100'] # 'celeba']
dataset_dir_lst = { 'cifar10_imb' : '/volumes1/datasets/cifar/CIFAR10',
                   }
# Define parameters
lst_arch = ['resnet18mam'] #'resnet18mam'
num_runs = 2
start_seed = 0
log_file = "../error_log.txt"

model_params = {
    "cifar10_imb" : {'lr': '0.1', 'epochs':'100', 'wd':'0.0005', 'batch_size':128},
}
modes = ["vlm"]

loss_types = ['sim']  # Example: ['kl']
loss_wt_lst = [20.0, 50.0] #, 15.0]
text_enc_lst = ['sent_transf']  # Example: ['bert']
gpt_path_lst = {
    "cifar10_imb": 'cl_datasets/metadata/cifar10_descriptions.json',
}
# New experiment settings
experiment_settings = [
    {"perc": 0.02, "gamma": -1, "corrupt_prob": 0.0},  # Setting 1
    {"perc": 0.05, "gamma": -1, "corrupt_prob": 0.0},  # Setting 1
    {"perc": 0.1, "gamma": -1, "corrupt_prob": 0.0},  # Setting 1
    {"perc": 0.2, "gamma": -1, "corrupt_prob": 0.0},  # Setting 1
    {"perc": 0.5, "gamma": -1, "corrupt_prob": 0.0},  # Setting 1
]
# Create a list of all combinations
combinations = list(itertools.product(
    modes,
    text_enc_lst,
    loss_types,
    loss_wt_lst,
    datasets,
    lst_arch,
    range(start_seed, start_seed + num_runs)
))

# Function to handle errors
def handle_error(exp_id, error_message):
    print(f"Error occurred in experiment {exp_id}: {error_message}")
    with open(log_file, "a") as f:
        f.write(f"Experiment ID: {exp_id}\nError Message: {error_message}\n\n")

# Iterate over combinations
for mode, text_enc, loss_mode, loss_wt, dataset, arch, seed in combinations:
    # Set model parameters
    lr = model_params[dataset]['lr']
    epochs = model_params[dataset]['epochs']
    batch_size = model_params[dataset]['batch_size']
    wd = model_params[dataset]['wd']
    dataset_dir = dataset_dir_lst[dataset]

    for setting in experiment_settings:
        perc = setting['perc']
        gamma = setting['gamma']
        corrupt_prob = setting['corrupt_prob']
        if mode == 'normal':
            exp_id = (
                f"{mode}-{arch}-{dataset}-p-{perc}-s-{seed}"
            )
        else:
            exp_id = (
                f"revproj-{mode}-{arch}-{dataset}-l-{loss_mode}-{loss_wt}-text-{text_enc}-p-{perc}-s-{seed}"
            )
        print(f"Running experiment {exp_id}")

        # Construct the command
        cmd = [
            "python", "/volumes1/vlm-cl/continual_VLM/main_normal.py",
            "--experiment_id", exp_id,
            "--model", "er",
            "--dataset", dataset,
            "--dataset_dir", dataset_dir,
            "--lr", lr,
            "--n_epochs", epochs,
            "--batch_size", "128",
            "--tensorboard", "1",
            "--nowand", "1",
            "--ignore_other_metrics", "1",
            "--wandb_project", "continual_VLM",
            "--wandb_entity", "sngowda42",
            "--output_dir", "/volumes1/vlm-cl/normal_cls/class_imb",
            "--arch", arch,
            "--scheduler", "cosine",
            "--seed", str(seed),
            "--optim_wd", wd,
            "--text_model", text_enc,
            "--loss_wt", str(loss_wt), str(loss_wt), str(loss_wt), str(loss_wt),
            "--loss_mode", loss_mode,
            '--gpt_path', gpt_path_lst[dataset],
            "--rev_proj",
            "--mode", mode,
            "--perc", str(perc),
            "--gamma", str(gamma),
            "--corrupt_prob", str(corrupt_prob)
        ]

        try:
            # Run the command
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            # Handle error with error message
            handle_error(exp_id, str(e))

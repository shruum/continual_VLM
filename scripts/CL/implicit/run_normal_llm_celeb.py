import subprocess
import os
import itertools


datasets = ['cifartint'] # 'celeba']
dataset_dir_lst = { 'cifar10' : '/volumes1/datasets/cifar/CIFAR10',
                    'cifartint': '/volumes1/datasets/cifar/CIFAR10',
                    'cifar100': '/volumes1/datasets/cifar/CIFAR100',
                   'celeba': '/volumes1/datasets/celeba'
                   }
# Define parameters
lst_arch = ['resnet18mamllm'] #'resnet18mam'
num_runs = 3
start_seed = 0
log_file = "../cls/error_log.txt"

model_params = {
    "cifar10" : {'lr': '0.1', 'epochs':'100', 'wd':'0.0005', 'batch_size':128},
    "cifar100": {'lr': '0.1', 'epochs': '100', 'wd': '0.0005', 'batch_size': 128},
    "cifartint": {'lr': '0.1', 'epochs': '100', 'wd': '0.0005', 'batch_size': 128},
    "celeba": {'lr': '0.1', 'epochs': '100', 'wd': '0.0005', 'batch_size': 128},
}
lr_lst = [0.001] #0.001, 0.01]
epoch_lst = [100]
llm_block = "sent_transf"
# wd_lst = [0.1, 0.001, 0.0001]

modes = [ "normal"] #["normal"] #, "vlm"]
# Create a list of all combinations
combinations = list(itertools.product(
    modes,
    lr_lst,
    epoch_lst,
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
for mode, lr, epochs, dataset, arch, seed in combinations:
    # Set model parameters
    # lr = model_params[dataset]['lr']
    # epochs = model_params[dataset]['epochs']
    batch_size = model_params[dataset]['batch_size']
    wd = model_params[dataset]['wd']
    dataset_dir = dataset_dir_lst[dataset]

    exp_id = f"ix-{mode}-{arch}-{llm_block}-{dataset}-desc--l{lr}-e-{epochs}-s-{seed}"
    print(f"Running experiment {exp_id}")
    # Construct the command
    cmd = [
        "python", "/volumes1/vlm-cl/continual_VLM/main_normal.py",
        "--experiment_id", exp_id,
        "--model", "er",
        "--dataset", dataset,
        "--dataset_dir", dataset_dir,
        "--lr", str(lr),
        "--n_epochs", str(epochs),
        "--batch_size", "128",
        "--tensorboard", "1",
        "--nowand", "1",
        "--ignore_other_metrics", "1",
        "--wandb_project", "continual_VLM",
        "--wandb_entity", "sngowda42",
        "--output_dir", "/volumes1/vlm-cl/dytox_cls",
        "--arch", arch,
        "--scheduler", "cosine",
        "--seed", str(seed),
        "--optim_wd", wd,
        "--mode", mode,
        "--llama",
        "--llm_block", llm_block
    ]

    try:
        # Run the command
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # Handle error with error message
        handle_error(exp_id, str(e))

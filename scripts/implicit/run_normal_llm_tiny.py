import subprocess
import os
import itertools


datasets = ['tinyimagenet']
dataset_dir_lst = { 'cifar10' : '/volumes1/datasets/cifar/CIFAR10',
                   'cifar100': '/volumes1/code-cls/InBiaseD/data',
                   'celeba': '/volumes1/datasets',
                   'tinyimagenet': '/volumes1/datasets/tiny-imagenet-200'
                   }
# Define parameters
lst_arch = ['resnet18mamllm'] #'resnet18mam'
num_runs = 1
start_seed = 42
log_file = "../cls/error_log.txt"
llm_block_lst = ['sent_transf'] #'clip',
model_params = {
    "cifar10" : {'lr': '0.1', 'epochs':'100', 'wd':'0.0005', 'batch_size':128},
    "cifar100": {'lr': '0.1', 'epochs': '100', 'wd': '0.0005', 'batch_size': 128},
    "tinyimagenet": {'lr': '0.03', 'epochs': '100', 'wd': '0.0005', 'batch_size': 128},
}
lr_lst = [0.0001, 0.001, 0.005]
wd_lst = [0.01] # 0.0005]
modes = [ "normal"] #["normal"] #, "vlm"]
# Create a list of all combinations
combinations = list(itertools.product(
    modes,
    lr_lst,
    # epoch_lst,
    wd_lst,
    datasets,
    lst_arch,
    llm_block_lst,
    range(start_seed, start_seed + num_runs)
))
def handle_error(exp_id, error_message):
    print(f"Error occurred in experiment {exp_id}: {error_message}")
    with open(log_file, "a") as f:
        f.write(f"Experiment ID: {exp_id}\nError Message: {error_message}\n\n")

# Iterate over combinations
for mode, lr, wd, dataset, arch, llm_block, seed in combinations:
    # Set model parameters
    # lr = model_params[dataset]['lr']
    epochs = model_params[dataset]['epochs']
    batch_size = model_params[dataset]['batch_size']
    # wd = model_params[dataset]['wd']
    dataset_dir = dataset_dir_lst[dataset]

    exp_id = f"impl-{mode}-{arch}-{llm_block}-{dataset}-l{lr}-e{epochs}-wd{wd}-s-{seed}"
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
        "--batch_size", str(batch_size),
        "--tensorboard", "1",
        "--nowand", "1",
        "--ignore_other_metrics", "1",
        "--wandb_project", "continual_VLM",
        "--wandb_entity", "sngowda42",
        "--output_dir", "/volumes1/vlm-cl/dytox_cls/implicit",
        "--arch", arch,
        "--scheduler", "cosine",
        "--seed", str(seed),
        "--optim_wd", str(wd),
        "--mode", mode,
        "--llm_block", llm_block,
        "--llama",
    ]

    try:
        # Run the command
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # Handle error with error message
        handle_error(exp_id, str(e))

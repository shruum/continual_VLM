import subprocess
import os
import itertools


datasets = ['cifar10_imb'] # 'cifar100'] # 'celeba']
dataset_dir_lst = { 'cifar10_imb' : '/volumes1/datasets/cifar/CIFAR10',
                   }
lst_arch = ['resnet18mamllm'] #'resnet18mam'
num_runs = 1
start_seed = 42
log_file = "../error_log.txt"
model_params = {
    "cifar10_imb" : {'lr': '0.1', 'epochs':'100', 'wd':'0.0005', 'batch_size':128},
}
lr_lst = [0.0001, 0.001, 0.005]
wd_lst = [0.01] # 0.0005]
modes = [ "normal"] #["normal"] #, "vlm"]
llm_block_lst = ['sent_transf'] #'clip',
experiment_settings = [
    {"perc": 0.05, "gamma": -1, "corrupt_prob": 0.0},
    {"perc": 0.1, "gamma": -1, "corrupt_prob": 0.0},
    {"perc": 0.2, "gamma": -1, "corrupt_prob": 0.0},
    {"perc": 0.5, "gamma": -1, "corrupt_prob": 0.0},
]
# Create a list of all combinations
combinations = list(itertools.product(
    modes,
    lr_lst,
    wd_lst,
    datasets,
    lst_arch,
    llm_block_lst,
    range(start_seed, start_seed + num_runs)
))

# Function to handle errors
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

    for setting in experiment_settings:
        perc = setting['perc']
        gamma = setting['gamma']
        corrupt_prob = setting['corrupt_prob']
        if mode == 'normal':
            exp_id = (
                f"impl-{mode}-{arch}-{llm_block}-{dataset}-l{lr}-e{epochs}-wd{wd}-p{perc}-s-{seed}"
            )
        else:
            pass
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
            "--output_dir", "/volumes1/vlm-cl/dytox_cls/implicit_classimb",
            "--arch", arch,
            "--scheduler", "cosine",
            "--seed", str(seed),
            "--optim_wd", str(wd),
            "--mode", mode,
            "--llm_block", llm_block,
            "--llama",
            "--perc", str(perc),
        ]

        try:
            # Run the command
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            # Handle error with error message
            handle_error(exp_id, str(e))

import subprocess
import os
import itertools

# Define parameters
lst_buffer_size = [200, 500]
lst_arch = ['resnet18mamllm'] #, 'resnet18mam']  # Including original and new architectures
num_runs = 1
start_seed = 42
datasets = ["seq-cifar10"] # "dn4il"]  # Expanded list to include new datasets
dataset_dir_lst = {
    "seq-cifar10" : "/volumes2/datasets/cifar/CIFAR10",
    "seq-cifar100": "/volumes2/datasets/cifar/CIFAR100",
    "seq-tinyimg" : "/volumes2/datasets/tiny-imagenet-200",
    "dn4il": "/volumes2/datasets/DN4IL"
}
output_dir = "/volumes1/vlm-cl/rebuttal"
log_file = os.path.join(output_dir, "error_log_method.txt")

# Additional parameters specific to 'ix' method
model_params = {
    'si': {'lr': '0.03', 'c': '0.5', 'xi': '1.0', 'epochs':'100', 'batch_size': 128},
    'ewc_on': {'lr': '0.03', 'e_lambda':'10', 'gamma':'1.0', 'epochs':'100', 'batch_size': 128},
}
lr_lst = [0.001, 0.01]
wd_lst = [0.0, 0.1, 0.01]
modes = ["normal", "vlm"]

# Iterate over combinations
combinations = list(itertools.product(
    modes,
    lr_lst,
    wd_lst,
    datasets,
    lst_arch,
    range(start_seed, start_seed + num_runs)
))

# Function to handle errors
def handle_error(exp_id):
    print(f"Error occurred in experiment {exp_id}")
    with open(log_file, "a") as f:
        f.write(f"{exp_id}\n")

# Processing the experiment setup
for mode, lr, wd, dataset, arch, seed in combinations:
    for model in model_params.keys():
        epochs = model_params[model]['epochs']  # Assuming 'ix' uses the same model_params index
        dataset_dir = dataset_dir_lst[dataset]
        for buffer_size in lst_buffer_size:
            exp_id = f"ix-{mode}-{model}-{arch}-{dataset}-lr{lr}-wd{wd}-e{epochs}-s{seed}"
            print(f"Running experiment {exp_id}")

            # Construct command
            cmd = [
                "python", "/volumes1/vlm-cl/continual_VLM/main.py",
                "--experiment_id", exp_id,
                "--model", model,  # Using the specific model
                "--dataset", dataset,
                "--dataset_dir", dataset_dir,
                "--lr", str(lr),
                "--n_epochs", str(epochs),
                "--batch_size", str(model_params[model]['batch_size']),
                "--tensorboard", "1",
                "--nowand", "1",
                "--ignore_other_metrics", "1",
                "--wandb_project", "continual_VLM",
                "--wandb_entity", "sngowda42",
                "--output_dir", output_dir,
                "--arch", arch,
                "--scheduler", "cosine",
                "--seed", str(seed),
                "--optim_wd", str(wd),
                "--mode", mode,
                "--llm_block", 'sent_transf',  # Specific LLM block
                "--llama",
            ]

            if model == 'bic':
                cmd.append("--buffer_size")
                cmd.append(str(buffer_size))
            elif model == 'er':
                cmd.append("--buffer_size")
                cmd.append(str(buffer_size))
                minibatch_size = model_params[model]['minibatch_size']
                cmd.append("--minibatch_size")
                cmd.append(str(minibatch_size))
            elif model == 'si':
                c = model_params[model]['c']
                xi = model_params[model]['xi']
                cmd.append("--c")
                cmd.append(c)
                cmd.append("--xi")
                cmd.append(xi)
            elif model == 'ewc_on':
                e_lambda = model_params[model]['e_lambda']
                gamma = model_params[model]['gamma']
                cmd.append("--e_lambda")
                cmd.append(e_lambda)
                cmd.append("--gamma")
                cmd.append(gamma)

            try:
                # Execute command
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                handle_error(exp_id)
                print(f"Command failed with error: {e}")
            except Exception as e:
                handle_error(exp_id)
                print(f"Unexpected error: {e}")


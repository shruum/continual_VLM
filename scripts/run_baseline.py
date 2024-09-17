import subprocess
import os
import itertools

# Define parameters
lst_buffer_size = [200, 500]  # Example: [100, 200, 500]
lst_arch = ['resnet18mam'] #'resnet18mam'
num_runs = 1
start_seed = 42
datasets = ["seq-tinyimg"] #"seq-cifar10", "dn4il", ] #"seq-cifar10",
dataset_dir_lst = {
    "seq-cifar10" : "/volumes1/datasets/cifar",
    "seq-tinyimg" : "/volumes1/datasets/tiny-imagenet-200",
    "dn4il": "/volumes1/datasets/DN4IL"
}
log_file = "error_log.txt"

model_params = {
    "seq-tinyimg" : {
        200: {'er': {'lr': '0.1', 'epochs':'100', 'alpha': None, 'beta': None, 'c': None, 'xi': None, 'minibatch_size': 32},
                 # 'der': {'lr': '0.03', 'alpha': '0.3', 'beta': None, 'c': None, 'xi': None, 'minibatch_size': 32},
                 # 'derpp': {'lr': 0.03, 'alpha': '0.1', 'beta': '0.5', 'c': None, 'xi': None, 'minibatch_size': 32},
                 # 'si': {'lr': '0.03', 'alpha': None, 'beta': None, 'c': '0.5', 'xi': '1.0', 'minibatch_size': None}
             },
        500: {'er': {'lr': '0.03', 'epochs':'100', 'alpha': None, 'beta': None, 'c': None, 'xi': None, 'minibatch_size': 32},
        },
    },
    "seq-cifar10": {
        'er': {'lr': '0.1', 'epochs': '50', 'alpha': None, 'beta': None, 'c': None, 'xi': None, 'minibatch_size': 32},
        # 'der': {'lr': '0.03', 'alpha': '0.3', 'beta': None, 'c': None, 'xi': None, 'minibatch_size': 32},
        # 'derpp': {'lr': 0.03, 'alpha': '0.1', 'beta': '0.5', 'c': None, 'xi': None, 'minibatch_size': 32},
        # 'si': {'lr': '0.03', 'alpha': None, 'beta': None, 'c': '0.5', 'xi': '1.0', 'minibatch_size': None}
        },
    "dn4il": {
        'er': {'lr': '0.1', 'epochs': '50', 'alpha': None, 'beta': None, 'c': None, 'xi': None, 'minibatch_size': 32},
        # 'der': {'lr': '0.03', 'alpha': '0.3', 'beta': None, 'c': None, 'xi': None, 'minibatch_size': 32},
        # 'derpp': {'lr': 0.03, 'alpha': '0.1', 'beta': '0.5', 'c': None, 'xi': None, 'minibatch_size': 32},
        # 'si': {'lr': '0.03', 'alpha': None, 'beta': None, 'c': '0.5', 'xi': '1.0', 'minibatch_size': None}
    },
}

# Create a list of all combinations
combinations = list(itertools.product(
    lst_buffer_size,
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
for buffer_size, dataset, arch, seed in combinations:
# buffer_size = None
# for dataset, arch, seed in combinations:
    for model in model_params[dataset].keys():
        # Set model parameters
        alpha = model_params[dataset][model]['alpha']
        beta = model_params[dataset][model]['beta']
        c = model_params[dataset][model]['c']
        xi = model_params[dataset][model]['xi']
        lr = model_params[dataset][model]['lr']
        epochs = model_params[dataset][model]['epochs']
        minibatch_size = model_params[dataset][model]['minibatch_size']
        dataset_dir = dataset_dir_lst[dataset]

        exp_id = (
            f"{model}-l{lr}-{arch}-{dataset}-buf-{buffer_size}-s-{seed}"
        )
        print(f"Running experiment {exp_id}")

        # Construct the command
        cmd = [
            "python", "/volumes1/vlm-cl/continual_VLM/main.py",
            "--experiment_id", exp_id,
            "--model", model,
            "--dataset", dataset,
            "--dataset_dir", dataset_dir,
            "--lr", lr,
            "--n_epochs", epochs,
            "--batch_size", "32",
            "--tensorboard", "1",
            "--nowand", "1",
            "--ignore_other_metrics", "1",
            "--wandb_project", "continual_VLM",
            "--wandb_entity", "sngowda42",
            "--output_dir", "/volumes1/vlm-cl/baseline_final",
            "--arch", arch,
            "--seed", str(seed),
        ]
        # Add model-specific arguments
        # Add model-specific arguments
        if buffer_size is not None:
            cmd.append("--buffer_size")
            cmd.append(str(buffer_size))
        if minibatch_size is not None:
            cmd.append("--minibatch_size")
            cmd.append(str(minibatch_size))
        if alpha is not None:
            cmd.append("--alpha")
            cmd.append(alpha)
        if beta is not None:
            cmd.append("--beta")
            cmd.append(beta)
        if c is not None:
            cmd.append("--c")
            cmd.append(c)
        if xi is not None:
            cmd.append("--xi")
            cmd.append(xi)

        try:
            # Run the command
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            # Handle error with error message
            handle_error(exp_id, str(e))

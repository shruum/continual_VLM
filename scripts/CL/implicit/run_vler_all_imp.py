import subprocess
import os
import itertools

# Define parameters
lst_buffer_size = [200, 500]  # Example: [100, 200, 500]
lst_arch = ['resnet18mamllm']
num_runs = 1
start_seed = 42
datasets = ["seq-cifar100"] #, "seq-cifar100", "seq-tinyimg"]
gpt_path_lst = {
    "seq-cifar10": 'cl_datasets/metadata/cifar10_descriptions.json',
    "seq-cifar100": 'cl_datasets/metadata/cifar100_descriptions.json',
    "seq-tinyimg": 'cl_datasets/metadata/tinyimagenet_description.json',
}
dataset_dir_lst = {
    "seq-cifar10" : "/volumes1/datasets/cifar/CIFAR10",
    "seq-cifar100": "/volumes1/datasets/cifar/CIFAR100",
    "seq-tinyimg" : "/volumes1/datasets/tiny-imagenet-200",
}

log_file = "error_log_method.txt"
model_params = {
    'seq-cifar10': {'lr': '0.1', 'epochs':'100', 'wd': 0.01, 'batch_size': 128, 'minibatch_size': 32},
    'seq-cifar100': {'lr': '0.1', 'epochs': '100', 'wd': 0.01, 'batch_size': 128, 'minibatch_size': 32},

}
lr_lst = [0.001, 0.005]
wd_lst = [0.01, 0.0005]
modes = [ "normal"] #["normal"] #, "vlm"]
llm_block = 'sent_transf' #'clip']
model ='er'
n_tasks_cif=10
# Create a list of all combinations
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

# Iterate over combinations
for mode, lr, wd, dataset, arch, seed in combinations:
    for buffer_size in lst_buffer_size if model != 'vl_si' else [None]:

        epochs = model_params[dataset]['epochs']
        # wd = model_params[dataset]['wd']
        minibatch_size = model_params[dataset]['minibatch_size']
        dataset_dir = dataset_dir_lst[dataset]
        exp_id = (
            f"lgix-{model}-{arch}-{dataset}{n_tasks_cif}-b-{buffer_size}--l{lr}-w{wd}-e-{epochs}-s-{seed}"
        )
        print(f"Running experiment {exp_id}")

        # Construct the command
        cmd = [
            "python", "/volumes1/vlm-cl/continual_VLM/main.py",
            "--experiment_id", exp_id,
            "--model", model,
            "--dataset", dataset,
            "--dataset_dir", dataset_dir,
            "--lr", str(lr),
            "--n_epochs", str(epochs),
            "--batch_size", "32",
            "--tensorboard", "1",
            "--nowand", "1",
            "--ignore_other_metrics", "1",
            "--wandb_project", "continual_VLM",
            "--wandb_entity", "sngowda42",
            "--output_dir", "/volumes1/vlm-cl/dytox_cls/cl",
            "--arch", arch,
            "--scheduler", "cosine",
            "--seed", str(seed),
            "--optim_wd", str(wd),
            "--mode", mode,
            "--buffer_size", str(buffer_size),
            "--minibatch_size", str(minibatch_size),
            "--llm_block", llm_block,
            "--llama",
            "--n_tasks_cif", str(n_tasks_cif)
        ]


        try:
            # Run the command
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            # Handle error
            handle_error(exp_id)
            print(f"Command failed with error: {e}")
        except Exception as e:
            # Handle any other unexpected errors
            handle_error(exp_id)
            print(f"Unexpected error: {e}")



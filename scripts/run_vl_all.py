import subprocess
import os
import itertools

# Define parameters
lst_buffer_size = [200, 500]  # Example: [100, 200, 500]
lst_arch = ['resnet18mam'] # 'resnet50mam']
# lst_lr = [0.03]
num_runs = 1
start_seed = 42
datasets = ["seq-tinyimg"] #["seq-cifar10", "dn4il", "seq-tinyimg"]
loss_types = ['sim']  # Example: ['kl']
loss_wt_lst = [6.0, 10.0, 14.0] #, 15.0]
text_enc_lst = ['sent_transf']  # Example: ['bert']
gpt_path_lst = {
    "seq-cifar10": 'cl_datasets/metadata/cifar10_descriptions.json',
    "seq-tinyimg": '/volumes1/datasets/tinyimagenet_description.json',
    "dn4il": '/volumes1/datasets/domainnet_description_100.json'
}
dataset_dir_lst = {
    "seq-cifar10" : "/volumes1/datasets/cifar",
    "seq-tinyimg" : "/volumes1/datasets/tiny-imagenet-200",
    "dn4il": "/volumes1/datasets/DN4IL"
}

log_file = "error_log_method.txt"

model_params = {
    'vl_er': {'lr': '0.1', 'epochs':'100', 'alpha': None, 'beta': None, 'c': None, 'xi': None, 'buffer_size': 200, 'minibatch_size': 32},
    'vl_der': {'lr': '0.03', 'epochs':'100','alpha': '0.3', 'beta': None, 'c': None, 'xi': None, 'minibatch_size': 32},
    # 'vl_derpp': {'lr': '0.03', 'alpha': '0.1', 'beta': '0.5', 'c': None, 'xi': None, 'minibatch_size': 32},
    # 'vl_si': {'lr': '0.03', 'alpha': None, 'beta': None, 'c': '0.5', 'xi': '1.0', 'minibatch_size': None}
}

# Create a list of all combinations
combinations = list(itertools.product(
    # lst_lr,
    text_enc_lst,
    loss_types,
    loss_wt_lst,
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
for text_enc, loss_mode, loss_wt, dataset, arch, seed in combinations:
    for model in model_params.keys():
        for buffer_size in lst_buffer_size if model != 'vl_si' else [None]:
            alpha = model_params[model]['alpha']
            beta = model_params[model]['beta']
            c = model_params[model]['c']
            xi = model_params[model]['xi']
            lr = model_params[model]['lr']
            epochs = model_params[model]['epochs']
            minibatch_size = model_params[model]['minibatch_size']
            dataset_dir = dataset_dir_lst[dataset]

            exp_id = (
                f"revproj-{model}-{arch}-{dataset}-desc-{buffer_size}--e-{epochs}-l-{loss_mode}-{loss_wt}-text-{text_enc}-s-{seed}"
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
                "--text_model", text_enc,
                "--loss_wt", str(loss_wt), str(loss_wt), str(loss_wt), str(loss_wt),
                "--ignore_other_metrics", "1",
                "--wandb_project", "continual_VLM",
                "--wandb_entity", "sngowda42",
                "--output_dir", "/volumes1/vlm-cl/results_final",
                "--loss_mode", loss_mode,
                '--gpt_path', gpt_path_lst[dataset],
                "--rev_proj",
                "--arch", arch,
                "--seed", str(seed),
            ]
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
                # Handle error
                handle_error(exp_id)
                print(f"Command failed with error: {e}")
            except Exception as e:
                # Handle any other unexpected errors
                handle_error(exp_id)
                print(f"Unexpected error: {e}")
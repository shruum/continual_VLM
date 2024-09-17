import subprocess
import os
import itertools

# Define parameters
lst_buffer_size = [200]  # Example: [100, 200, 500]
lst_lr = [0.03]
num_runs = 1
start_seed = 0
datasets = ["dn4il"]
loss_types = ['sim', 'kl']  # Example: ['kl']
loss_wt_lst = [2.0, 6.0, 12.0]
text_enc_lst = ['sent_transf']  # Example: ['bert']
gpt_path_lst = {
    "seq-cifar10": 'cl_datasets/metadata/cifar10_descriptions.json',
    "dn4il": '/volumes1/datasets/domainnet_description_100.json'
}
log_file = "error_log.txt"

model_params = {
    'vl_er': {'alpha': None, 'beta': None, 'c': None, 'xi': None},
    'vl_derpp': {'alpha': '0.1', 'beta': '0.5', 'c': None, 'xi': None},
    'vl_si': {'alpha': None, 'beta': None, 'c': '0.5', 'xi': '1.0'}
}

# Create a list of all combinations
combinations = list(itertools.product(
    lst_buffer_size,
    lst_lr,
    text_enc_lst,
    loss_types,
    loss_wt_lst,
    datasets,
    range(start_seed, start_seed + num_runs)
))

# Function to handle errors
def handle_error(exp_id):
    print(f"Error occurred in experiment {exp_id}")
    with open(log_file, "a") as f:
        f.write(f"{exp_id}\n")

# Iterate over combinations
for buffer_size, lr, text_enc, loss_mode, loss_wt, dataset, seed in combinations:
    for model in model_params.keys():
        alpha = model_params[model]['alpha']
        beta = model_params[model]['beta']
        c = model_params[model]['c']
        xi = model_params[model]['xi']

        exp_id = (
            f"{model}-{dataset}-desc-{buffer_size}--lr-{lr}-l-{loss_mode}-{loss_wt}-text-{text_enc}-s-{seed}"
        )
        print(f"Running experiment {exp_id}")

        # Construct the command
        cmd = [
            "python", "/volumes1/vlm-cl/continual_VLM/main.py",
            "--experiment_id", exp_id,
            "--model", model,
            "--dataset", dataset,
            "--dataset_dir", "/volumes1/datasets/DN4IL",
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
            "--output_dir", "/volumes1/vlm-cl/results_cu11",
            "--loss_mode", loss_mode,
            '--gpt_path', gpt_path_lst[dataset]
        ]
        # Add model-specific arguments
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
        except subprocess.CalledProcessError:
            # Handle error
            handle_error(exp_id)

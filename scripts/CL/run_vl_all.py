import subprocess
import os
import itertools

# Define parameters
lst_buffer_size = [200, 500]
buffer_size = 200
lst_lr = [0.1, 0.03]
lst_arch = ['resnet18mam'] # 'resnet50mam']
num_runs = 1
start_seed = 42
datasets = ["dn4il"] #, "seq-tinyimg"] # "dn4il"] #, "seq-cifar10"]
loss_types = ['sim']  # Example: ['kl']
loss_wt_lst = [5.0, 15.0, 30.0] #, 15.0]
epochs = [50]
text_enc_lst = ['sent_transf'] #'clip']  # Example: ['bert']
gpt_path_lst = {
    "seq-cifar10": 'cl_datasets/metadata/cifar10_descriptions.json',
    # "seq-tinyimg": 'cl_datasets/metadata/tinyimagenet_description.json',
    "dn4il": 'cl_datasets/metadata/domainnet_description_100.json'
}
dataset_dir_lst = {
    "seq-cifar10" : "/volumes2/datasets/cifar",
    "seq-tinyimg" : "/volumes2/datasets/tiny-imagenet-200",
    "dn4il": "/volumes2/datasets/DN4IL"
}
output_dir = "/volumes1/vlm-cl/rebuttal"
log_file = os.path.join(output_dir, "error_log_method.txt")

model_params = {
    # 'vl_er': {'lr': '0.1', 'epochs':'100', 'alpha': None, 'beta': None, 'c': None, 'xi': None, 'minibatch_size': 32},
    'vl_si': {'lr': '0.03', 'c': '0.5', 'xi': '1.0'},
    'vl_ewc_on': {'lr': '0.03', 'e_lambda':'10', 'gamma':'1.0'},
    # 'vl_bic': {'lr': '0.03', 'minibatch_size': None}
}

# Create a list of all combinations
combinations = list(itertools.product(
    lst_lr,
    epochs,
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
for lr, epochs, text_enc, loss_mode, loss_wt, dataset, arch, seed in combinations:
    for model in model_params.keys():
        # for buffer_size in lst_buffer_size if model == 'vl_bic' else [None]:
        dataset_dir = dataset_dir_lst[dataset]
        # lr = model_params[model]['lr']

        if model == 'vl_er' or model == 'vl_bic':
            exp_id = f"ex-{model}-{arch}-{dataset}-buf-{buffer_size}--e-{epochs}-l{lr}-{loss_wt}-text-{text_enc}-s-{seed}"
        else:
            exp_id = f"ex-{model}-{arch}-{dataset}-e-{epochs}-l{lr}-{loss_wt}-text-{text_enc}-s-{seed}"

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
            "--text_model", text_enc,
            "--ignore_other_metrics", "1",
            "--wandb_project", "continual_VLM",
            "--wandb_entity", "sngowda42",
            "--output_dir", output_dir,
            "--loss_mode", loss_mode,
            "--loss_wt", str(loss_wt), str(loss_wt), str(loss_wt), str(loss_wt),
            '--gpt_path', gpt_path_lst[dataset],
            "--rev_proj",
            "--arch", arch,
            "--seed", str(seed),
            "--save_model"
        ]
        # Add model-specific arguments
        if model == 'vl_bic':
            cmd.append("--buffer_size")
            cmd.append(str(buffer_size))
        elif model == 'vl_er':
            cmd.append("--buffer_size")
            cmd.append(str(buffer_size))
            minibatch_size = model_params[model]['minibatch_size']
            cmd.append("--minibatch_size")
            cmd.append(str(minibatch_size))
        elif model == 'vl_si':
            c = model_params[model]['c']
            xi = model_params[model]['xi']
            cmd.append("--c")
            cmd.append(c)
            cmd.append("--xi")
            cmd.append(xi)
        elif model == 'vl_ewc_on':
            e_lambda = model_params[model]['e_lambda']
            gamma = model_params[model]['gamma']
            cmd.append("--e_lambda")
            cmd.append(e_lambda)
            cmd.append("--gamma")
            cmd.append(gamma)
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



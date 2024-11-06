import subprocess
import os
import itertools


datasets = ['tinyimagenet'] # 'cifar100'] # 'celeba']
dataset_dir_lst = {'tinyimagenet': '/volumes1/datasets/tiny-imagenet-200'
                   }
# Define parameters
lst_arch = ['resnet18mam'] #'resnet18mam'
num_runs = 2
start_seed = 0
log_file = "error_log.txt"

model_params = {
    "tinyimagenet": {'lr': '0.03', 'epochs': '100', 'wd': '0.0005', 'batch_size': 128},
}
modes = ["vlm"] #,"normal"]

lst_lr = [0.05]
loss_types = ['sim']  # Example: ['kl']
loss_wt_lst = [200.0, 80.0]
text_enc_lst = ['sent_transf']  # Example: ['bert']
gpt_path_lst = {
    "cifar10": 'cl_datasets/metadata/cifar10_descriptions.json',
    "cifartint": 'cl_datasets/metadata/cifar10_descriptions.json',
    "cifar100": 'cl_datasets/metadata/cifar100_descriptions.json',
    "celeba": '/volumes1/datasets/celeba_description.json',
    "tinyimagenet": '/volumes1/vlm-cl/continual_VLM/cl_datasets/metadata/tinyimagenet_description.json',

}
# Create a list of all combinations
combinations = list(itertools.product(
    modes,
    lst_lr,
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
# for mode, text_enc, loss_mode, loss_wt, dataset, arch, seed in combinations:
for mode, lr, text_enc, loss_mode, loss_wt, dataset, arch, seed in combinations:
    # Set model parameters
    # lr = model_params[dataset]['lr']
    epochs = str(model_params[dataset]['epochs'])
    batch_size = str(model_params[dataset]['batch_size'])
    wd = str(model_params[dataset]['wd'])
    dataset_dir = dataset_dir_lst[dataset]

    exp_id = f"{mode}-{arch}-{dataset}-desc-e-{epochs}-l-{lr}-{loss_wt}-text-{text_enc}-s-{seed}"

    print(f"Running experiment {exp_id}")
    # Construct the command
    cmd = [
        "python", "/volumes1/vlm-cl/continual_VLM/main_normal.py",
        "--experiment_id", exp_id,
        "--model", "er",
        "--dataset", dataset,
        "--dataset_dir", dataset_dir,
        "--lr", str(lr),
        "--n_epochs", epochs,
        "--batch_size", "128",
        "--tensorboard", "1",
        "--nowand", "1",
        "--ignore_other_metrics", "1",
        "--wandb_project", "continual_VLM",
        "--wandb_entity", "sngowda42",
        "--output_dir", "/volumes1/vlm-cl/normal_cls",
        "--arch", arch,
        "--scheduler", "cosine",
        "--seed", str(seed),
        "--optim_wd", wd,
        "--text_model", text_enc,
        "--loss_wt", str(loss_wt), str(loss_wt), str(loss_wt), str(loss_wt),
        "--loss_mode", loss_mode,
        '--gpt_path', gpt_path_lst[dataset],
        "--mode", mode,
    ]
    try:
        # Run the command
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # Handle error with error message
        handle_error(exp_id, str(e))


# "--text_model", text_enc,
# "--loss_wt", str(loss_wt), str(loss_wt), str(loss_wt), str(loss_wt),
# "--loss_mode", loss_mode,
# '--gpt_path', gpt_path_lst[dataset],
# "--rev_proj",
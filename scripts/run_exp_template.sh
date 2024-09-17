#!/bin/bash -l
module load 2023
module load Anaconda3/2023.07-2
module load Python/3.11.3-GCCcore-12.3.0

#SBATCH --time=1-23:45:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Create the Conda environment from the YAML file if it doesn't already exist
if ! conda info --envs | grep -q vlm_cl; then
  conda env create -f /home/snarasimhe/workspace/continual_VLM/env.yaml
fi

source activate vlm_cl

python main.py \
    --experiment_id "$exp_id" \
    --model vl_er \
    --dataset "$dataset" \
    --dataset_dir "/home/snarasimhe/workspace/datasets/tiny-imagenet-200" \
    --buffer_size "$buffer_size" \
    --lr 0.1 \
    --n_epochs 1 \
    --minibatch_size 32 \
    --batch_size 32 \
    --tensorboard 1 \
    --nowand 0 \
    --text_model "$text_enc" \
    --loss_wt "$loss_wt" "$loss_wt" "$loss_wt" "$loss_wt" \
    --ignore_other_metrics 1 \
    --wandb_project continual_VLM \
    --wandb_entity sngowda42 \
    --loss_mode "$loss_mode"
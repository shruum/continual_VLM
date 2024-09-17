#!/bin/bash -l

printf -v date '%(%Y-%m-%d_%H-%M-%S)T' -2
echo "Experiment started at $date"

#SBATCH -p gpu
#SBATCH --time=1-23:45:00
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4

/home/snarasimhe/miniconda3/bin/conda init
conda activate vlm_cl
/home/snarasimhe/miniconda3/envs/vlm_cl/bin/python3 main.py \
--experiment_id temp \
--model vl_er \
--dataset seq-tinyimg \
--buffer 200 \
--lr 0.1 \
--minibatch_size 32 \
--batch_size 32 \
--n_epochs 1 \
--nowand 1 \
--wandb_project continual_VLM \
--wandb_entity sngowda42 \
--text_model bert \
--dataset_dir /home/snarasimhe/workspace/datasets/tiny-imagenet-200 \
--ignore_other_metrics 1 \
--output_dir /experiments \


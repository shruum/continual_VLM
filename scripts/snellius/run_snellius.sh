#!/bin/bash -l

printf -v date '%(%Y-%m-%d_%H-%M-%S)T' -2
echo "Experiment started at $date"

#SBATCH --time=1-23:45:00
#SBATCH --partition=gpu_mig
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9

conda activate vlm_cl

python main.py --model er \
--dataset seq-tinyimg \
--buffer 200 \
--lr 0.1 \
--minibatch_size 32 \
--batch_size 32 \
--n_epochs 50 \
--nowand 1 \
--dataset_dir /scratch-local/snarasimhe/datasets/tiny-imagenet-200 \
--ignore_other_metrics 1 \
--output_dir /experiments \

#--console_output "/home/bgrooten01/console_logs/neurl/neurotrailsRL_${date}_job-id${SLURM_JOBID}.txt" \
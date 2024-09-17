#!/bin/bash

# Define parameter values
text_enc_lst=('sent_transf' 'bert')

# Loop over parameter combinations and submit Slurm jobs
for text_model in "${text_enc_lst[@]}"; do
  exp_id="vl-er-res50-${text_model}"
  echo "Submitting job for combination: $exp_id"
  # Create a temporary script file
  tmp_script=$(mktemp /home/snarasimhe/workspace/continual_VLM/tmp/slurm_script.XXXXXX)
  cat <<EOF > "$tmp_script"
#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --time=1-23:45:00
#SBATCH --cpus-per-task=18

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_cl
/home/snarasimhe/miniconda3/envs/vlm_cl/bin/python3 main.py \
  --experiment_id $exp_id \
  --model vl_er \
  --dataset seq-tinyimg \
  --dataset_dir /home/snarasimhe/workspace/datasets/tiny-imagenet-200 \
  --buffer_size 200 \
  --lr 0.1 \
  --n_epochs 1 \
  --minibatch_size 32 \
  --batch_size 32 \
  --tensorboard 0 \
  --nowand 1 \
  --text_model $text_model \
  --loss_wt 1 1 1 1 \
  --ignore_other_metrics 1 \
  --wandb_project continual_VLM \
  --wandb_entity sngowda42 \
  --output_dir experiments \
  --loss_mode l2
EOF
  # Submit the temporary script
  sbatch "$tmp_script"
  rm "$tmp_script"

done
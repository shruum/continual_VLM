#!/bin/bash
# Define the datasets you want to run
datasets=("seq-cifar10")
declare -A gpt_path_lst
gpt_path_lst["seq-cifar10"]='cl_datasets/metadata/cifar10_descriptions.json'

# Define the models and their parameters
declare -A model_params
model_params["vl_er"]="--buffer_size 200 --minibatch_size 32"
# Define the list of buffer sizes (only used for vl_er and vl_derpp)
buffer_sizes=(200)

# Define other parameters
text_enc_lst=('sent_transf')
loss_mode='sim'
loss_wt_lst=(14.0 50.0)
loss_loc_lst=('before')
lst_lr=(0.05)
num_runs=2
start_seed=42

# Loop over all combinations
for dataset in "${datasets[@]}"; do
  for buffer_size in "${buffer_sizes[@]}"; do
    for model in "${!model_params[@]}"; do
      for text_enc in "${text_enc_lst[@]}"; do
        for lr in "${lst_lr[@]}"; do
          for loss_wt in "${loss_wt_lst[@]}"; do
            for loss_loc in "${loss_loc_lst[@]}"; do
              for seed in $(seq $start_seed $((start_seed + num_runs - 1))); do
                exp_id="${model}-${dataset}-b${buffer_size}-lr-${lr}-l-${loss_loc}-${loss_wt}-${text_enc}-s-${seed}"
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
#SBATCH -o /home/snarasimhe/workspace/continual_VLM/slurm/cif100/slurm-%j.out
#SBATCH -e /home/snarasimhe/workspace/continual_VLM/slurm/cif100/slurm-%j.err
# Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_cl

# Run the Python script with the current parameters
/home/snarasimhe/miniconda3/envs/vlm_cl/bin/python3 main.py \
  --experiment_id $exp_id \
  --model $model \
  --arch resnet18mam \
  --dataset $dataset \
  --dataset_dir /home/snarasimhe/workspace/datasets/CIFAR10 \
  --lr $lr \
  --n_epochs 50 \
  --batch_size 32 \
  --tensorboard 0 \
  --nowand 1 \
  --text_model $text_enc \
  --loss_wt $loss_wt $loss_wt $loss_wt $loss_wt \
  --ignore_other_metrics 1 \
  --wandb_project continual_VLM \
  --wandb_entity sngowda42 \
  --output_dir /home/snarasimhe/workspace/continual_VLM/results_final \
  --loss_mode $loss_mode \
  --gpt_path ${gpt_path_lst[$dataset]} \
  --seed $seed \
  --rev_proj \
  --loss_loc $loss_loc \
  --buffer_size $buffer_size \
  --minibatch_size 32 \
EOF
                #Submit the temporary script
                sbatch "$tmp_script"
	              rm "$tmp_script"
              done
            done
          done
        done
      done
    done
  done
done
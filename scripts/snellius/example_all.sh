#!/bin/bash
# Define the norm_datasets you want to run
norm_datasets=("seq-cifar10")
gpt_path_lst["seq-cifar10"]='/home/snarasimhe/workspace/norm_datasets/cifar10_descriptions.json'
# Define the models and their parameters
declare -A model_params
model_params["vl_er"]="--buffer_size 200 --minibatch_size 32"
model_params["vl_derpp"]="--buffer_size 200 --minibatch_size 32 --alpha 0.1 --beta 0.5"
model_params["vl_si"]="--c 0.5 --xi 1.0"

# Define the list of buffer sizes (only used for vl_er and vl_derpp)
buffer_sizes=(200)
# Define other parameters
text_enc_lst=('sent_transf', 'bert')
loss_types=('sim' 'kl')
loss_wt_lst=(2.0 6.0 12.0)
lst_lr=(0.03)
num_runs=1
start_seed=0
# Loop over all combinations
for dataset in "${norm_datasets[@]}"; do
  for buffer_size in "${buffer_sizes[@]}"; do
    for model in "${!model_params[@]}"; do
      for text_enc in "${text_enc_lst[@]}"; do
        for loss_mode in "${loss_types[@]}"; do
          for loss_wt in "${loss_wt_lst[@]}"; do
            for lr in "${lst_lr[@]}"; do
              for seed in $(seq $start_seed $((start_seed + num_runs - 1))); do

                # Create a unique experiment ID
                exp_id="${model}-${dataset}-desc-${buffer_size}--lr-${lr}-l-${loss_mode}-${loss_wt}-text-${text_enc}-s-${seed}"

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
#SBATCH -o ${output_dir}/%j_"${model}"_"${text_enc}"_"${dataset}"_"${loss_mode}"_"${lr}"_"${loss_wt}"_"$(date +%m-%d-%H-%M)".out
# Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_cl

# Run the Python script with the current parameters
/home/snarasimhe/miniconda3/envs/vlm_cl/bin/python3 main.py \
  --experiment_id $exp_id \
  --model $model \
  --dataset $dataset \
  --dataset_dir /home/snarasimhe/workspace/datasets \
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
  --output_dir /home/snarasimhe/workspace/continual_VLM/results \
  --loss_mode $loss_mode \
  --gpt_path ${gpt_path_lst[$dataset]} \
  ${model_params[$model]}

EOF
                # Submit the temporary script
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
#!/bin/bash


lst_buffer_size=(200) #500) #[100, 200, 500]
num_runs=1
start_seed=0
dataset="seq-tinyimg"
loss_types=('l2' 'nce')
loss_wt_lst=(5.0 10.0 20.0)
text_enc_lst=('sent_transf' 'bert') #'clip'
log_file="error_log.txt"

# Function to handle errors
handle_error() {
  echo "Error occurred in experiment $exp_id"
  echo "$exp_id" >> "$log_file"
}

# Iterate over parameters
for buffer_size in "${lst_buffer_size[@]}"; do
  for text_enc in "${text_enc_lst[@]}"; do
    for loss_mode in "${loss_types[@]}"; do
      for loss_wt in "${loss_wt_lst[@]}"; do
          for seed in $(seq $start_seed $(($start_seed + $num_runs - 1))); do
              exp_id="vl-er-res50-${buffer_size}-loss-${loss_mode}-${loss_wt}-text-${text_enc}-s-${seed}"
              echo "Running experiment $exp_id"
              {
              python /volumes1/vlm-cl/continual_VLM/main.py \
                  --experiment_id "$exp_id" \
                  --model vl_er \
                  --dataset "$dataset" \
                  --dataset_dir /volumes1/datasets/tiny-imagenet-200 \
                  --buffer_size "$buffer_size" \
                  --lr 0.1 \
                  --n_epochs 50 \
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
              } || handle_error
          done
      done
    done
  done
done
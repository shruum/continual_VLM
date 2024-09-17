  GNU nano 2.9.8                                                                                  run_snellius_multi.sh

#!/bin/bash -l

printf -v date '%(%Y-%m-%d_%H-%M-%S)T' -2
echo "Experiment started at $date"

# Parameters
lst_buffer_size=(200) #500) #[100, 200, 500]
num_runs=1
start_seed=0
dataset="seq-tinyimg"
loss_types=('l2') # 'nce' 'kl' 'sim')
loss_wt_lst=(5.0) # 10.0 20.0)
text_enc_lst=('sent_transf') # 'bert') #'clip'
log_file="error_log.txt"

# Function to handle errors
handle_error() {
  echo "Error occurred in experiment $exp_id"
  echo "$exp_id" >> "$log_file"
}

# Iterate over parameters and submit jobs
for buffer_size in "${lst_buffer_size[@]}"; do
  for text_enc in "${text_enc_lst[@]}"; do
    for loss_mode in "${loss_types[@]}"; do
      for loss_wt in "${loss_wt_lst[@]}"; do
        for seed in $(seq $start_seed $(($start_seed + $num_runs - 1))); do
          exp_id="snel-vl-er-res50-${buffer_size}-loss-${loss_mode}-${loss_wt}-text-${text_enc}-s-${seed}"
          echo "Submitting experiment $exp_id"

          # Create a job script with specific parameters
          job_script="run_exp_${exp_id}.sh"
          cp run_exp_template.sh $job_script
          sed -i "s/\$exp_id/$exp_id/g" $job_script
          sed -i "s/\$dataset/$dataset/g" $job_script
          sed -i "s/\$buffer_size/$buffer_size/g" $job_script
          sed -i "s/\$text_enc/$text_enc/g" $job_script
          sed -i "s/\$loss_mode/$loss_mode/g" $job_script
          sed -i "s/\$loss_wt/$loss_wt/g" $job_script

          # Submit the job
          sbatch $job_script
        done
      done
    done
  done
done


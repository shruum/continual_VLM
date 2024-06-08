python main.py  \
--experiment_id vqa \
--model vqa_er \
--dataset seq-cifar10 \
--buffer_size 200 \
--lr 0.1 \
--n_epochs 50 \
--minibatch_size 32 \
--batch_size 32 \
--output_dir results \
--loss_type l2 \
--nowand 1 \
# --loss_wt {loss_wt} {loss_wt} {loss_wt} {loss_wt} \
# --tensorboard 1 \
# --nowand 1 \

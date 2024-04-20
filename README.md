# Continual Learning using Vision Language Tasks

## Running

```
buffer_size = 200 (or 500)
python main.py 
    --experiment_id vl_er_exp \
    --seed 0 \
    --model vl_er \
    --dataset seq-cifar10 \
    --buffer_size {buffer_size} \
    --lr 0.1 \
    --n_epochs 50 \
    --batch_size 32 \
    --minibatch_size 32 \
    --output_dir /output/ \
    --loss_mode l2 \ (or nce)
    --loss_wt 1 1 1 1 \
    --tensorboard \
    --csv_log \
```


## Setup

+ Use `./utils/main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters from the paper.
+ New models can be added to the `models/` folder.
+ New datasets can be added to the `datasets/` folder.

## Models

+ eXtended-DER (X-DER)

+ Dark Experience Replay (DER)
+ Dark Experience Replay++ (DER++)

+ Learning a Unified Classifier Incrementally via Rebalancing (LUCIR)
+ Greedy Sampler and Dumb Learner (GDumb)
+ Bias Correction (BiC)
+ Regular Polytope Classifier (RPC)

+ Gradient Episodic Memory (GEM)
+ A-GEM
+ A-GEM with Reservoir (A-GEM-R)
+ Experience Replay (ER)
+ Meta-Experience Replay (MER)
+ Function Distance Regularization (FDR)
+ Greedy gradient-based Sample Selection (GSS)
+ Hindsight Anchor Learning (HAL)
+ Incremental Classifier and Representation Learning (iCaRL)
+ online Elastic Weight Consolidation (oEWC)
+ Synaptic Intelligence
+ Learning without Forgetting
+ Progressive Neural Networks

## Datasets

**Class-Il / Task-IL settings**

+ Sequential MNIST
+ Sequential CIFAR-10
+ Sequential Tiny ImageNet
+ Sequential CIFAR-100

**Domain-IL settings**

+ Permuted MNIST
+ Rotated MNIST

**General Continual Learning setting**

+ MNIST-360


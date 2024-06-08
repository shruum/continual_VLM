# Guidelines
- Make minimal changes in the main.py and training scripts
- Use begin task and end task attributes in the method to log additional models
- Eval mode assumes that the model only returns the logits

# Changes
- Added arguments for experiment id and output directory

### Loggers.py

- Added input arguments: output_directory, experiment_id, model_idt
- Changed output format from pyd to csv
- Added taskwise performance file in write method
- Output format for multiple models would be logged as:

```
-<Output dir>
  ├── results
  │   ├── class-il
  ├── ├──├── <method>
  ├── ├──├── ├── <experiment_id>
  ├── ├──├── ├── ├── logs.csv
  ├── ├──├── ├── ├── logs_plastic_model.csv
  ├── ├──├── ├── ├── logs_stable_model.csv
  ├── ├──├── ├── ├── task_performance.csv
  ├── ├──├── ├── ├── task_performance_stable_model.csv
  ├── ├──├── ├── ├── task_performance_plastic_model.csv
  ├── ├──├── ├── ├── model.ph
  │   ├── task-il
  ├── ├──├── <method>
  ├── ├──├── ├── <experiment_id>
  ├── ├──├── ├── ├── logs.csv
  ├── ├──├── ├── ├── logs_plastic_model.csv
  ├── ├──├── ├── ├── logs_stable_model.csv

```

### args

- Added experiment_id, output_dir and dataset_dir in experiment_args

### training.py
- Added tensorboard logging

### Continual Model

* Added task variable in init
* Added methods:
  - init_loggers: initializing loggers for addit models
  - save_models: saving the model's state dict
  - evaluate: evaluate function with eval_model param
  - eval_before_training: evaluate addit model on the current task before training
  - eval_addit_models: Evaluate additional models on all the tasks trained so far
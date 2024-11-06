import pandas as pd
import os
import re
import csv

base_dir = '/volumes1/vlm-cl/fahad/results_cuda2_22_09/class-il/seq-cifar100/vl_er/20tsk'

# Define the new header for the output
new_header_10 = [
    'accmean_task1', 'accmean_task2', 'accmean_task3', 'accmean_task4', 'accmean_task5',
    'accmean_task6', 'accmean_task7', 'accmean_task8', 'accmean_task9', 'accmean_task10',
    'accuracy_1_task1', 'accuracy_1_task2', 'accuracy_2_task2', 'accuracy_1_task3', 'accuracy_2_task3', 'accuracy_3_task3',
    'accuracy_1_task4', 'accuracy_2_task4', 'accuracy_3_task4', 'accuracy_4_task4', 'accuracy_1_task5', 'accuracy_2_task5', 'accuracy_3_task5',
    'accuracy_4_task5', 'accuracy_5_task5', 'accuracy_1_task6', 'accuracy_2_task6', 'accuracy_3_task6', 'accuracy_4_task6', 'accuracy_5_task6',
    'accuracy_6_task6', 'accuracy_1_task7', 'accuracy_2_task7', 'accuracy_3_task7', 'accuracy_4_task7', 'accuracy_5_task7', 'accuracy_6_task7',
    'accuracy_7_task7', 'accuracy_1_task8', 'accuracy_2_task8', 'accuracy_3_task8', 'accuracy_4_task8', 'accuracy_5_task8', 'accuracy_6_task8',
    'accuracy_7_task8', 'accuracy_8_task8', 'accuracy_1_task9', 'accuracy_2_task9', 'accuracy_3_task9', 'accuracy_4_task9', 'accuracy_5_task9',
    'accuracy_6_task9', 'accuracy_7_task9', 'accuracy_8_task9', 'accuracy_9_task9', 'accuracy_1_task10', 'accuracy_2_task10', 'accuracy_3_task10',
    'accuracy_4_task10', 'accuracy_5_task10', 'accuracy_6_task10', 'accuracy_7_task10', 'accuracy_8_task10', 'accuracy_9_task10', 'accuracy_10_task10',
    'forward_transfer', 'backward_transfer', 'forgetting', 'seed', 'notes', 'non_verbose', 'disable_log', 'tensorboard', 'validation', 'ignore_other_metrics',
    'debug_mode', 'nowand', 'wandb_entity', 'wandb_project', 'save_model', 'experiment_id', 'dataset_dir', 'output_dir', 'dataset', 'mnist_seed', 'n_tasks_cif',
    'n_tasks_mnist', 'deg_inc', 'model', 'arch', 'lr', 'optim_wd', 'optim_mom', 'optim_nesterov', 'scheduler', 'n_epochs', 'batch_size', 'llama', 'llama_path',
    'mode', 'device', 'buffer_size', 'minibatch_size', 'aux', 'img_size', 'loss_type', 'loss_wt', 'dir_aux', 'buf_aux', 'rev_proj', 'ser', 'aug_prob',
    'data_combine', 'loss_mode', 'text_model', 'ser_weight', 'gpt_path', 'use_lr_scheduler', 'lr_steps', 'conf_jobnum', 'conf_timestamp', 'conf_host'
]

new_header_20 = [
    'accmean_task1', 'accmean_task2', 'accmean_task3', 'accmean_task4', 'accmean_task5', 'accmean_task6',
    'accmean_task7', 'accmean_task8', 'accmean_task9', 'accmean_task10', 'accmean_task11', 'accmean_task12',
    'accmean_task13', 'accmean_task14', 'accmean_task15', 'accmean_task16', 'accmean_task17', 'accmean_task18',
    'accmean_task19', 'accmean_task20', 'accuracy_1_task1', 'accuracy_1_task2', 'accuracy_2_task2',
    'accuracy_1_task3', 'accuracy_2_task3', 'accuracy_3_task3', 'accuracy_1_task4', 'accuracy_2_task4',
    'accuracy_3_task4', 'accuracy_4_task4', 'accuracy_1_task5', 'accuracy_2_task5', 'accuracy_3_task5',
    'accuracy_4_task5', 'accuracy_5_task5', 'accuracy_1_task6', 'accuracy_2_task6', 'accuracy_3_task6',
    'accuracy_4_task6', 'accuracy_5_task6', 'accuracy_6_task6', 'accuracy_1_task7', 'accuracy_2_task7',
    'accuracy_3_task7', 'accuracy_4_task7', 'accuracy_5_task7', 'accuracy_6_task7', 'accuracy_7_task7',
    'accuracy_1_task8', 'accuracy_2_task8', 'accuracy_3_task8', 'accuracy_4_task8', 'accuracy_5_task8',
    'accuracy_6_task8', 'accuracy_7_task8', 'accuracy_8_task8', 'accuracy_1_task9', 'accuracy_2_task9',
    'accuracy_3_task9', 'accuracy_4_task9', 'accuracy_5_task9', 'accuracy_6_task9', 'accuracy_7_task9',
    'accuracy_8_task9', 'accuracy_9_task9', 'accuracy_1_task10', 'accuracy_2_task10', 'accuracy_3_task10',
    'accuracy_4_task10', 'accuracy_5_task10', 'accuracy_6_task10', 'accuracy_7_task10', 'accuracy_8_task10',
    'accuracy_9_task10', 'accuracy_10_task10', 'accuracy_1_task11', 'accuracy_2_task11', 'accuracy_3_task11',
    'accuracy_4_task11', 'accuracy_5_task11', 'accuracy_6_task11', 'accuracy_7_task11', 'accuracy_8_task11',
    'accuracy_9_task11', 'accuracy_10_task11', 'accuracy_11_task11', 'accuracy_1_task12', 'accuracy_2_task12',
    'accuracy_3_task12', 'accuracy_4_task12', 'accuracy_5_task12', 'accuracy_6_task12', 'accuracy_7_task12',
    'accuracy_8_task12', 'accuracy_9_task12', 'accuracy_10_task12', 'accuracy_11_task12', 'accuracy_12_task12',
    'accuracy_1_task13', 'accuracy_2_task13', 'accuracy_3_task13', 'accuracy_4_task13', 'accuracy_5_task13',
    'accuracy_6_task13', 'accuracy_7_task13', 'accuracy_8_task13', 'accuracy_9_task13', 'accuracy_10_task13',
    'accuracy_11_task13', 'accuracy_12_task13', 'accuracy_13_task13', 'accuracy_1_task14', 'accuracy_2_task14',
    'accuracy_3_task14', 'accuracy_4_task14', 'accuracy_5_task14', 'accuracy_6_task14', 'accuracy_7_task14',
    'accuracy_8_task14', 'accuracy_9_task14', 'accuracy_10_task14', 'accuracy_11_task14', 'accuracy_12_task14',
    'accuracy_13_task14', 'accuracy_14_task14', 'accuracy_1_task15', 'accuracy_2_task15', 'accuracy_3_task15',
    'accuracy_4_task15', 'accuracy_5_task15', 'accuracy_6_task15', 'accuracy_7_task15', 'accuracy_8_task15',
    'accuracy_9_task15', 'accuracy_10_task15', 'accuracy_11_task15', 'accuracy_12_task15', 'accuracy_13_task15',
    'accuracy_14_task15', 'accuracy_15_task15', 'accuracy_1_task16', 'accuracy_2_task16', 'accuracy_3_task16',
    'accuracy_4_task16', 'accuracy_5_task16', 'accuracy_6_task16', 'accuracy_7_task16', 'accuracy_8_task16',
    'accuracy_9_task16', 'accuracy_10_task16', 'accuracy_11_task16', 'accuracy_12_task16', 'accuracy_13_task16',
    'accuracy_14_task16', 'accuracy_15_task16', 'accuracy_16_task16', 'accuracy_1_task17', 'accuracy_2_task17',
    'accuracy_3_task17', 'accuracy_4_task17', 'accuracy_5_task17', 'accuracy_6_task17', 'accuracy_7_task17',
    'accuracy_8_task17', 'accuracy_9_task17', 'accuracy_10_task17', 'accuracy_11_task17', 'accuracy_12_task17',
    'accuracy_13_task17', 'accuracy_14_task17', 'accuracy_15_task17', 'accuracy_16_task17', 'accuracy_17_task17',
    'accuracy_1_task18', 'accuracy_2_task18', 'accuracy_3_task18', 'accuracy_4_task18', 'accuracy_5_task18',
    'accuracy_6_task18', 'accuracy_7_task18', 'accuracy_8_task18', 'accuracy_9_task18', 'accuracy_10_task18',
    'accuracy_11_task18', 'accuracy_12_task18', 'accuracy_13_task18', 'accuracy_14_task18', 'accuracy_15_task18',
    'accuracy_16_task18', 'accuracy_17_task18', 'accuracy_18_task18', 'accuracy_1_task19', 'accuracy_2_task19',
    'accuracy_3_task19', 'accuracy_4_task19', 'accuracy_5_task19', 'accuracy_6_task19', 'accuracy_7_task19',
    'accuracy_8_task19', 'accuracy_9_task19', 'accuracy_10_task19', 'accuracy_11_task19', 'accuracy_12_task19',
    'accuracy_13_task19', 'accuracy_14_task19', 'accuracy_15_task19', 'accuracy_16_task19', 'accuracy_17_task19',
    'accuracy_18_task19', 'accuracy_19_task19', 'accuracy_1_task20', 'accuracy_2_task20', 'accuracy_3_task20',
    'accuracy_4_task20', 'accuracy_5_task20', 'accuracy_6_task20', 'accuracy_7_task20', 'accuracy_8_task20',
    'accuracy_9_task20', 'accuracy_10_task20', 'accuracy_11_task20', 'accuracy_12_task20', 'accuracy_13_task20',
    'accuracy_14_task20', 'accuracy_15_task20', 'accuracy_16_task20', 'accuracy_17_task20', 'accuracy_18_task20',
    'accuracy_19_task20', 'accuracy_20_task20', 'forward_transfer', 'backward_transfer', 'forgetting', 'seed',
    'notes', 'non_verbose', 'disable_log', 'tensorboard', 'validation', 'ignore_other_metrics', 'debug_mode',
    'nowand', 'wandb_entity', 'wandb_project', 'save_model', 'experiment_id', 'dataset_dir', 'output_dir',
    'dataset', 'mnist_seed', 'n_tasks_cif', 'n_tasks_mnist', 'deg_inc', 'model', 'arch', 'lr', 'optim_wd',
    'optim_mom', 'optim_nesterov', 'scheduler', 'n_epochs', 'batch_size', 'llama', 'llama_path', 'mode',
    'device', 'buffer_size', 'minibatch_size', 'aux', 'img_size', 'loss_type', 'loss_wt', 'dir_aux', 'buf_aux',
    'rev_proj', 'ser', 'aug_prob', 'data_combine', 'loss_mode', 'text_model', 'ser_weight', 'gpt_path',
    'use_lr_scheduler', 'lr_steps', 'conf_jobnum', 'conf_timestamp', 'conf_host'
]


# Check how many tasks are in each row
def get_num_tasks(row):
    return len([col for col in row.index if col.startswith('accmean_task')])


# Walk through all subdirectories in the base directory
for root, dirs, files in os.walk(base_dir):
    for dir in dirs:
        file = 'logs.csv'
        # Construct the full file path
        file_path = os.path.join(root, dir, file)
        # Open the file and read all lines
        with open(file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            lines = list(reader)
        # Check if there are at least 3 rows
        if len(lines) < 3:
            if len(lines) == 1:
                row_data = lines[0]
                row_df = pd.DataFrame([row_data])
                row_df.columns = new_header_20
                row_df.to_csv(file_path, index=False, header=True)
            print(f"File {file_path} has less than 3 rows, skipping.")
            continue
        # elif len(lines) == 3:
        #     # del lines[2]
        #     # row_data = lines
        #     # row_df = pd.DataFrame(row_data)
        #     # row_df.to_csv(file_path, index=False, header=False)
        #     # pass
        #     # Split the data into two sets (5-task and 10-task)
        #     third_row_data = lines[2]
        #     third_row_df = pd.DataFrame([third_row_data])
        #     third_row_df.columns = new_header_10
        #     pattern = r"revproj-vl_er-resnet18mam-seq-cifar100-\[5, 10, 20\]-"
        #     match = re.search(pattern, file_path)
        #     if match:
        #         extracted_name = match.group(0)
        #         new_name = extracted_name.replace("[5, 10, 20]", "10")
        #         new_file_path = file_path.replace(extracted_name, new_name)
        #     else:
        #         print("Pattern not found in the file path.")
        #     os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        #     # Save the ten_task_row DataFrame with the new header to a new Excel file
        #     third_row_df.to_csv(new_file_path, index=False, header=False)
        #
        #     del lines[2]
        #     row_data = lines
        #     row_df = pd.DataFrame(row_data)
        #     row_df.to_csv(file_path, index=False, header=False)
        # elif len(lines) == 4:
        #
        #     third_row_data = lines[2]
        #     third_row_df = pd.DataFrame([third_row_data])
        #     third_row_df.columns = new_header_10
        #     pattern = r"revproj-vl_er-resnet18mam-seq-cifar100-\[5, 10, 20\]-"
        #     match = re.search(pattern, file_path)
        #     if match:
        #         extracted_name = match.group(0)
        #         new_name = extracted_name.replace("[5, 10, 20]", "10")
        #         new_file_path = file_path.replace(extracted_name, new_name)
        #     else:
        #         print("Pattern not found in the file path.")
        #     os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        #     # Save the ten_task_row DataFrame with the new header to a new Excel file
        #     third_row_df.to_csv(new_file_path, index=False, header=False)
        #
        #
        #     fourth_row_data = lines[3]
        #     fourth_row_df = pd.DataFrame([fourth_row_data])
        #     fourth_row_df.columns = new_header_20
        #     pattern = r"revproj-vl_er-resnet18mam-seq-cifar100-\[5, 10, 20\]-"
        #     match = re.search(pattern, file_path)
        #     if match:
        #         extracted_name = match.group(0)
        #         new_name = extracted_name.replace("[5, 10, 20]", "20")
        #         new_file_path = file_path.replace(extracted_name, new_name)
        #     else:
        #         print("Pattern not found in the file path.")
        #     os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        #     # Save the ten_task_row DataFrame with the new header to a new Excel file
        #     fourth_row_df.to_csv(new_file_path, index=False, header=False)
        #
        #     del lines[2]
        #     del lines[2]
        #     row_data = lines
        #     row_df = pd.DataFrame(row_data)
        #     row_df.to_csv(file_path, index=False, header=False)


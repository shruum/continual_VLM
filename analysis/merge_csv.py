import os
import pandas as pd

def merge_csv_files(base_folder, output_file):
    # Initialize a list to hold dataframes
    dfs = []

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.csv'):
                # Read each CSV file
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                dfs.append(df)

    # Concatenate all dataframes
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        # Write the merged dataframe to a CSV file
        merged_df.to_csv(output_file, index=False)
        print(f"Merged CSV saved to {output_file}")
    else:
        print("No CSV files found.")

# Define the base folder containing the subfolders with CSV files
base_folder = '/volumes1/vlm-cl/continual_VLM/experiments/results/class-il/seq-tinyimg/vl_er'
# Define the output file path
output_file = os.path.join(base_folder, 'merged_output.csv')
merge_csv_files(base_folder, output_file)

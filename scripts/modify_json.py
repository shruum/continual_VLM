import json
import os

# Define the input and output file paths'
input_filepath = '/volumes1/datasets/tinyimagenet_description.json'
output_filepath = '/volumes1/datasets/tinyimagenet_description.json'

# Load JSON data from the input file
with open(input_filepath, 'r') as json_file:
    data = json.load(json_file)

# Remove newline characters from each value in the dictionary
cleaned_data = {key: value.replace('\n', '') for key, value in data.items()}

# Save the modified data back to a JSON file
with open(output_filepath, 'w') as json_file:
    json.dump(cleaned_data, json_file, indent=4)

print(f"Newline characters removed and saved to '{output_filepath}'")

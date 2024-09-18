import pickle

# Path to the CIFAR-100 dataset meta file
meta_file = '/volumes1/datasets/cifar/CIFAR100/cifar-100-python/meta'

# Load the CIFAR-100 meta file
with open(meta_file, 'rb') as file:
    meta_data = pickle.load(file, encoding='latin1')

# Extract the fine label names (class names) and their corresponding IDs (0 to 99)
fine_label_names = meta_data['fine_label_names']  # List of class names
fine_labels = list(range(len(fine_label_names)))  # List of class IDs (0 to 99)

# Print class names with their IDs
class_mapping = {i: fine_label_names[i] for i in fine_labels}
print(class_mapping)

# Save to a JSON file if needed
import json
output_file = '/volumes1/datasets/cifar/CIFAR100/cifar100_class_mapping.json'
with open(output_file, 'w') as json_file:
    json.dump(class_mapping, json_file, indent=4)

print(f"Class mapping saved to {output_file}")

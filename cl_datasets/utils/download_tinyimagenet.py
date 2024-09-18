import os
import requests
import zipfile

# Define the URL and local path
url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
output_path = "data/tiny-imagenet-200.zip"
dataset_dir = "data/tiny-imagenet-200"

# Download the dataset
def download_tiny_imagenet(url, output_path):
    print(f"Downloading Tiny ImageNet dataset from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(output_path, "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)
    print(f"Download completed and saved to {output_path}.")

# Extract the dataset
def extract_tiny_imagenet(zip_path, extract_path):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extraction completed. Dataset available at {extract_path}.")

# Check if the dataset is already downloaded and extracted
if not os.path.exists(output_path):
    download_tiny_imagenet(url, output_path)
else:
    print(f"{output_path} already exists. Skipping download.")

if not os.path.exists(dataset_dir):
    extract_tiny_imagenet(output_path, ".")
else:
    print(f"{dataset_dir} already exists. Skipping extraction.")

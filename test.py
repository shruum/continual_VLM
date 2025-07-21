import sys
import os
import csv

#mammoth_path = '/volumes1/vlm-cl/continual_VLM'
#print(mammoth_path)
#os.chdir(mammoth_path)
#sys.path.append(mammoth_path)
#sys.path.append(mammoth_path + '/norm_datasets')
#sys.path.append(mammoth_path + '/backbone')
#sys.path.append(mammoth_path + '/models')

from norm_datasets.dataset import DATASETS
from utils.normal_training import eval
from backbone.ResNet_mam import resnet18mam
from backbone.ResNet_mam_llm import resnet18mamllm
from backbone.ResNet import resnet18
from backbone.vit import *
from backbone.vit_llm import *
from backbone.ResNet_llm import *
from argparse import ArgumentParser
import datetime
import uuid
import socket


def parse_args():
    parser = ArgumentParser(description='Evaluation script', allow_abbrev=False)

    # Add your arguments here
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use for evaluation')
    parser.add_argument('--dataset_dir', type=str, default='./data', help='Path to the dataset directory')
    parser.add_argument('--arch', type=str, default='resnet18mam', help='Architecture (resnet18mam or resnet18mamllm)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save evaluation results')
    parser.add_argument('--model_path', type=str, help='Path to the model checkpoint')
    parser.add_argument('--llama', action='store_true', help='Use resnet18mamllm architecture')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for evaluation (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--llm_block', type=str, default='sent_transf')


    return parser.parse_args()


def load_model(args, dataset, device):
    # Load the model architecture based on the specified argument
    if args.arch == 'resnet18mamllm':
        backbone = resnet18mamllm(dataset.NUM_CLASSES).to(device)
    elif args.arch == 'resnet18mam':
        backbone = resnet18mam(dataset.NUM_CLASSES).to(device)
    elif args.arch == 'resnet18':
        backbone = resnet18(dataset.NUM_CLASSES).to(device)
    elif args.arch == 'vitsmall':
        backbone = vitsmall(dataset.NUM_CLASSES).to(device)
    elif args.arch == 'vitsmallllm':
        backbone = vitsmallllm(dataset.NUM_CLASSES, args.llm_block).to(device)
    elif args.arch == "resnet18llm":
        backbone = resnet18llm(dataset.NUM_CLASSES, 64, args.llm_block).to(device)

    # Load the saved model weights
    model_path = args.model_path
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        backbone.load_state_dict(checkpoint['state_dict'])
        print(f'Model loaded from {model_path}')
    else:
        raise FileNotFoundError(f'Checkpoint {model_path} not found.')

    return backbone


def save_results_to_csv(args, test_loss, test_accuracy):
    # Save the evaluation results to a CSV file
    result_file = os.path.join(args.output_dir, 'eval_results.csv')
    os.makedirs(args.output_dir, exist_ok=True)

    fieldnames = ['Experiment ID', 'Dataset', 'Test Loss', 'Test Accuracy', 'Date']
    values = [args.model_path, args.dataset, test_loss, test_accuracy * 100, str(datetime.datetime.now())]

    # Write results to the CSV file
    if os.path.exists(result_file):
        with open(result_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(values)
    else:
        with open(result_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)  # Write header
            writer.writerow(values)


def main():
    args = parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Detect device (CUDA or CPU)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    args.output_dir = os.path.dirname(args.model_path)
    # Load dataset
    dataset = DATASETS[args.dataset](args.dataset_dir)

    # Load test dataset and dataloader
    transform_test = dataset.transform_test
    testset = dataset.get_dataset('test') #, transform_test, transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load model
    model = load_model(args, dataset, device)

    # Evaluate the model
    test_loss, test_accuracy, _ = eval(model, device, test_loader, args)

    # Print evaluation results
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")

    # Save results to CSV
    save_results_to_csv(args, test_loss, test_accuracy)


if __name__ == '__main__':
    main()

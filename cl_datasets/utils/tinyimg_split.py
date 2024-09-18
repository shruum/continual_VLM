import os
import shutil
import torch
import subprocess
from torchvision import transforms
from utils.imgfolder import random_split, ImageFolderTrainVal, create_dir, attempt_move

def download_dset(path):
    create_dir(path)

    if not os.path.exists(os.path.join(path, 'tiny-imagenet-200.zip')):
        subprocess.call(
            "wget -P {} http://cs231n.stanford.edu/tiny-imagenet-200.zip".format(path),
            shell=True)
        print("Successfully downloaded TinyImgnet dataset.")
    else:
        print("Already downloaded TinyImgnet dataset in {}".format(path))

    if not os.path.exists(os.path.join(path, 'tiny-imagenet-200')):
        subprocess.call(
            "unzip {} -d {}".format(os.path.join(path, 'tiny-imagenet-200.zip'), path),
            shell=True)
        print("Successfully extracted TinyImgnet dataset.")
    else:
        print("Already extracted TinyImgnet dataset in {}".format(os.path.join(path, 'tiny-imagenet-200')))


def preprocess_val(root_path):
    """
    Uses val_annotations.txt to construct ImageFolder like structure.
    Images in 'image' folder are moved into class-folder.
    """
    val_path = os.path.join(root_path, 'val')
    annotation_path = os.path.join(val_path, 'val_annotations.txt')

    lines = [line.rstrip('\n') for line in open(annotation_path)]
    for line in lines:
        subs = line.split('\t')
        imagename = subs[0]
        dirname = subs[1]
        this_class_dir = os.path.join(val_path, dirname, 'images')
        if not os.path.isdir(this_class_dir):
            os.makedirs(this_class_dir)

        attempt_move(os.path.join(val_path, 'images', imagename), this_class_dir)


def divide_into_tasks(root_path, task_count=10):
    """
    Divides the Tiny Imagenet dataset into task_count tasks.
    Each task will have a subset of classes from the dataset.
    """
    print("Dividing into tasks...")
    nb_classes_task = 200 // task_count
    assert 200 % nb_classes_task == 0, "Total 200 classes must be divisible by number of classes per task"

    file_path = r"cl_datasets/metadata/wnids.txt"
    lines = [line.rstrip('\n') for line in open(file_path)]
    assert len(lines) == 200, "Should have 200 classes, but {} lines in wnids.txt".format(len(lines))
    subsets = ['train', 'val']
    img_paths = {t: {s: [] for s in subsets + ['classes', 'class_to_idx']} for t in range(1, task_count + 1)}

    for subset in subsets:
        task = 1
        for initial_class in range(0, 200, nb_classes_task):
            classes = lines[initial_class:initial_class + nb_classes_task]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            if len(img_paths[task]['classes']) == 0:
                img_paths[task]['classes'].extend(classes)
            img_paths[task]['class_to_idx'] = class_to_idx

            # Make subset dataset directory for each task
            for class_id, class_name in enumerate(classes):
                src_path = os.path.join(root_path, subset, class_name, 'images')
                imgs = [(os.path.join(src_path, f), class_to_idx[class_name]) for f in os.listdir(src_path)
                        if os.path.isfile(os.path.join(src_path, f))]
                img_paths[task][subset].extend(imgs)

            task += 1

    return img_paths


def create_train_val_test_imagefolder_dict(dataset_root, img_paths, task_count, outfile, no_crop=True, transform=False):
    """
    Create the dataset dictionary for each task and save it.
    """
    for task in range(1, task_count + 1):
        print("\nTASK ", task)

        # Normalization transform
        normalize = transforms.Normalize(mean=[0.4802, 0.4480, 0.3975], std=[0.2770, 0.2691, 0.2821])

        # Training dataset
        train_transf = transforms.Compose([
            transforms.RandomResizedCrop(56),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = ImageFolderTrainVal(
            dataset_root, None, transform=train_transf,
            classes=img_paths[task]['classes'], class_to_idx=img_paths[task]['class_to_idx'],
            imgs=img_paths[task]['train']
        )

        # Validation dataset
        val_transf = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(56),
            transforms.ToTensor(),
            normalize
        ])

        test_dataset = ImageFolderTrainVal(
            dataset_root, None, transform=val_transf,
            classes=img_paths[task]['classes'], class_to_idx=img_paths[task]['class_to_idx'],
            imgs=img_paths[task]['val']
        )

        # Save datasets
        out_dir = os.path.join(dataset_root, "{}tasks".format(task_count))
        create_dir(os.path.join(out_dir, str(task)))
        torch.save({'train': train_dataset, 'test': test_dataset}, os.path.join(out_dir, str(task), outfile))

        print("Saved datasets for Task {}: train={}, val={}".format(task, len(train_dataset), len(test_dataset)))


if __name__ == "__main__":
    dataset_root = '/volumes1/datasets/tiny-imagenet-200'  # Replace with your actual dataset root path
    task_count = 10  # Number of tasks to divide the dataset into

    # download_dset(dataset_root)
    preprocess_val(dataset_root)
    img_paths = divide_into_tasks(dataset_root, task_count=task_count)
    create_train_val_test_imagefolder_dict(dataset_root, img_paths, task_count, 'tinyimgnet_dataset.pth')

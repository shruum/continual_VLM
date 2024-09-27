import os
import csv
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD, Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


def save_results_normal(args, file, test_loss, test_accuracy, seed=0, mu=0, sigma = 0):

    names = [
        "exp",
        "mode",
        "seed",
        "lr",
        "epochs",
        "dataset",
        "network",
        "test_loss",
        "test_acc",
        "mu",
        "sigma",
        "model_dir",
    ]

    values = [
        args.experiment_id,
        args.mode,
        seed,
        args.lr,
        args.n_epochs,
        args.dataset,
        args.arch,
        test_loss,
        test_accuracy * 100,
        mu,
        sigma,
        os.path.join(args.experiment_id, 'checkpoints'),
    ]

    folder = os.path.dirname(file)
    os.makedirs(folder, exist_ok=True)

    if os.path.isfile(file):
        with open(file, "a") as f:
            writer = csv.writer(f)
            writer.writerow(values)
    else:
        np.savetxt(file, (names, values), delimiter=",", fmt="%s")

def adjust_learning_rate(epoch, epoch_steps, epoch_decay, optimizer):
    """decrease the learning rate"""

    if epoch in epoch_steps:
        current_lr = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = current_lr * epoch_decay
        print('=' * 60 + '\nChanging learning rate to %g\n' % (current_lr * epoch_decay) + '=' * 60)


def eval(model, device, data_loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)

    accuracy = correct / len(data_loader.dataset)
    return loss, accuracy, correct

def train_normal(args, dataset, model):

    transform_train = dataset.transform_train
    transform_test = dataset.transform_test
    # load dataset
    trainset = dataset.get_dataset('train', transform_train, transform_test)
    testset = dataset.get_dataset('test', transform_train, transform_test)
    #data
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)

    if args.arch == 'resnet18mamllm':
        optimizer = AdamW(model.backbone.parameters(), lr=args.lr, betas=(0.9,0.98), eps=1e-6,weight_decay=args.optim_wd) #lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    else:
        optimizer = SGD(model.backbone.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.optim_wd)


    scheduler = None
    if args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.epoch_step, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    print('*' * 60 + '\nTraining Mode: %s\n' % args.mode + '*' * 60)
    for epoch in tqdm(range(1, args.n_epochs + 1), desc='training epochs'):
        # adjust learning rate for SGD
        if scheduler:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, args.epoch_step, args.lr_decay_ratio, optimizer)

        model.train_normal(train_loader, optimizer, epoch)
    # get final test accuracy
    test_loss, test_accuracy, correct = eval(model.backbone, model.device, test_loader)
    save_results_normal(args, os.path.join(args.output_dir, args.experiment_id, 'results.csv'),
                                 test_loss, test_accuracy)

    save_dict = {
        'state_dict': model.backbone.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(save_dict, os.path.join(args.output_dir, args.experiment_id, f'model.ph'))

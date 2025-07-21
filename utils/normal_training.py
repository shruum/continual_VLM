import os
import csv
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD, Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import glob
import wandb

# try:
#     import wandb
#     wandb.login(key='aea8c8f148fd1c9feba13c1ecacb37ef951e6c05')
# except ImportError or AttributeError:
#     wandb = None

def get_latest_checkpoint(output_dir, experiment_id):
    # List all checkpoint files matching the pattern checkpoint_[epoch].pth
    checkpoint_files = glob.glob(os.path.join(output_dir, experiment_id, 'checkpoint_*.pth'))
    if not checkpoint_files:
        return None
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return checkpoint_files[-1]  # Return the latest checkpoint by epoch

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


def eval(model, device, data_loader, args=None):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            if args.arch == 'clip_vit':
                output = model(data)
                # output, _ = out.pooler_output, out.last_hidden_state.mean(dim=1)
            elif 'vit' in args.arch : #vittiny' or args.arch == 'vittinyllm' or args.arch == 'vitsmall' or args.arch == 'vitsmallllm':
                output, _ = model(data)
            else:
                output = model(data)

            loss += F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)

    accuracy = correct / len(data_loader.dataset)
    return loss, accuracy, correct

def train_normal(args, dataset, model):
    
    if not args.nowand:
        # assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        print(args.wandb_project)
        print(args.wandb_entity)
        # wandb.login(key="aea8c8f148fd1c9feba13c1ecacb37ef951e6c05")
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        args.wandb_url = wandb.run.get_url()

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

    if 'llm' in args.arch or "vit" in args.arch:
        optimizer = AdamW(model.backbone.parameters(), lr=args.lr, weight_decay=args.optim_wd)
    else:
        optimizer = SGD(model.backbone.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.optim_wd)
    print("Optimizer:", optimizer.__class__.__name__)

    scheduler = None
    if args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.epoch_step, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    print('*' * 60 + '\nTraining Mode: %s\n' % args.mode + '*' * 60)
    start_epoch = 1
    checkpoint_path = get_latest_checkpoint(args.output_dir, args.experiment_id)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.backbone.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in tqdm(range(start_epoch, args.n_epochs + 1), desc='training epochs'):
        # adjust learning rate for SGD
        if scheduler:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, args.epoch_step, args.lr_decay_ratio, optimizer)

        model.train_normal(train_loader, optimizer, epoch)

        if dataset.__class__.__name__ == 'Imagenet1k' and epoch % 5 == 0:
            checkpoint_data = {
                'state_dict': model.backbone.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            epoch_checkpoint_path = os.path.join(args.output_dir, args.experiment_id, f'checkpoint_{epoch}.pth')
            os.makedirs(os.path.join(args.output_dir, args.experiment_id), exist_ok=True)
            torch.save(checkpoint_data, epoch_checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")
        
        if not args.nowand and epoch % args.log_interval == 0:
            # evaluate on test set
            test_loss, test_accuracy, correct = eval(model.backbone, model.device, test_loader, args)
            print(f'Epoch {epoch}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')
            if not args.nowand:
                wandb.log({"Epoch": epoch, "test_loss": test_loss, "test_acc": test_accuracy * 100})

    # get final test accuracy
    test_loss, test_accuracy, correct = eval(model.backbone, model.device, test_loader, args)
    save_results_normal(args, os.path.join(args.output_dir, args.experiment_id, 'results.csv'),
                                 test_loss, test_accuracy)

    save_dict = {
        'state_dict': model.backbone.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(save_dict, os.path.join(args.output_dir, args.experiment_id, f'model.ph'))

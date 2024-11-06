import torch
import pandas as pd
from glob import glob
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from utils.adv_loss import eval_adv_robustness

cuda = torch.cuda.is_available()

# =============================================================================
# Load Datasets
# =============================================================================

dataset = 'seq-cifar10'
dataset_path = 'data/'

# =============================================================================
# Transform
# =============================================================================
# CIFAR 10
TRANSFORM = transforms.Compose(
    [
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2470, 0.2435, 0.2615))
     ]
)

# CIFAR 10
TRANSFORM_AUX = transforms.Compose(
    [
         transforms.ToTensor(),
     ]
)

dataset = CIFAR10('data', train=False, download=True, transform=TRANSFORM)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)
dataset_aux = CIFAR10('data', train=False, download=True, transform=TRANSFORM_AUX)
data_loader_aux = torch.utils.data.DataLoader(dataset_aux, batch_size=32, shuffle=False, num_workers=1)
info = {}
info["mean"] = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.float32)
info["std"] = torch.tensor((0.2470, 0.2435, 0.2615), dtype=torch.float32)

lst_methods = {
    'der': '/data/output-ai/fahad.sarfraz/SYNERgy/derpp_cifar10/task_models/seq-cifar10/derpp-cifar10-200-param-v1/task_5_model.ph',
    'er': '/data/output-ai/fahad.sarfraz/SYNERgy/er_cifar10/task_models/seq-cifar10/er-cifar10-200-param-v1-r3/task_5_model.ph',
    'cls-er': '/data/output-ai/fahad.sarfraz/SYNERgy/analysis_models/cifar10/cil_er/task_models/seq-cifar10/er-c10-200-param-v1-s-0/task_5_stable_model.ph',
    'aux_ema': '/data/output-ai/shruthi.gowda/continual/clmm_derpp_v2n1/results/class-il/seq-cifar10/derpp_mm_eman1_model/derppema-n1m-200-seq-cifar10-kl1.0-0.1-up0.2-s2_ema_net1/ema_net1_final.pth',
    'aux': '/data/output-ai/shruthi.gowda/continual/clmm_derpp_v2n1/results/class-il/seq-cifar10/derpp_mm_eman1_model/derppema-n1m-200-seq-cifar10-kl1.0-0.1-up0.2-s2/net_final.pth'
}

lst_buffer_size = [200]
lst_pgd_steps = [20]
eps_lst = [0.0, 0.25/255, 0.5/255, 1/255, 2/255, 4/255]
device = 'cuda'

results = {
    'id': [],
    'method': [],
    'eps': [],
    'num_steps': [],
    'accuracy': [],
    'robustness': []
}

for buffer_size in lst_buffer_size:
    # buffer_size = 500
    print('=' * 50)
    print(f'Buffer Size = {buffer_size}')
    print('=' * 50)

    for method in lst_methods:

        print(method)
        model_path = lst_methods[method]
        model = torch.load(model_path).to(device)

        for eps in eps_lst:
            print(eps)
            for num_steps in lst_pgd_steps:
                test_acc, test_rob = eval_adv_robustness(
                    model,
                    data_loader_aux if 'aux' in method else data_loader,
                    eps,  # 0.007
                    num_steps,
                    0.003,
                    info,
                    random=True,
                    norm=False if 'aux' in method else True,
                    device='cuda' if cuda else 'cpu'
                )

                print('Accuracy:', test_acc)
                print(f'PGD-{num_steps}', test_rob)
                results['method'].append(method)
                results['id'].append(model_path)
                results['eps'].append(eps)
                results['num_steps'].append(num_steps)
                results['accuracy'].append(test_acc)
                results['robustness'].append(test_rob)

        df = pd.DataFrame(results)
        df.to_csv('/volumes2/continual_learning/paper/analysis/adv/adv_cif10_s20.csv')

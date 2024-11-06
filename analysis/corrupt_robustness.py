import os
os.sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../"))

import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from utils.evaluate import evaluate
from imagecorruptions import corrupt

cuda = torch.cuda.is_available()

class arg_class():
    def __init__(self, corrupt_name, corrupt_severity):
        self.corrupt_name = corrupt_name
        self.corrupt_severity = corrupt_severity

class ImageCorruptions:
    def __init__(self, args):
        self.severity = args.corrupt_severity
        self.corruption_name = args.corrupt_name

    def __call__(self, image, labels=None):

        image = np.array(image)
        cor_image = corrupt(image, corruption_name=self.corruption_name,
                        severity=self.severity)

        return Image.fromarray(cor_image)

# =============================================================================
# Load Datasets
# =============================================================================

dataset_path = 'data/'
corrupt_list = ['brightness', 'contrast', 'fog', 'frost', 'snow',
                'gaussian_noise', 'shot_noise', 'impulse_noise',
                'motion_blur', 'defocus_blur', 'glass_blur', 'zoom_blur', 'gaussian_blur',
                'pixelate', 'elastic_transform', 'jpeg_compression', 'speckle_noise', 'spatter', 'saturate']
# =============================================================================
# Transform
# =============================================================================
# CIFAR 10
def transform_test(args):
    TRANSFORM = transforms.Compose(
        [
            ImageCorruptions(args),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))
         ]
    )

    # CIFAR 10
    TRANSFORM_AUX = transforms.Compose(
        [
            ImageCorruptions(args),
            transforms.ToTensor(),
         ]
    )
    return TRANSFORM, TRANSFORM_AUX

dataset_name = 'seq-cifar100' # 'seq-cifar10'
lst_methods = {
    'seq-cifar10':{
        'der': '/data/output-ai/fahad.sarfraz/SYNERgy/derpp_cifar10/task_models/seq-cifar10/derpp-cifar10-200-param-v1/task_5_model.ph',
        'er': '/data/output-ai/fahad.sarfraz/SYNERgy/er_cifar10/task_models/seq-cifar10/er-cifar10-200-param-v1-r3/task_5_model.ph',
        'cls-er': '/data/output-ai/fahad.sarfraz/SYNERgy/analysis_models/cifar10/cil_er/task_models/seq-cifar10/er-c10-200-param-v1-s-0/task_5_stable_model.ph',
        'aux_ema': '/data/output-ai/shruthi.gowda/continual/clmm_derpp_v2n1/results/class-il/seq-cifar10/derpp_mm_eman1_model/derppema-n1m-200-seq-cifar10-kl1.0-0.1-up0.2-s2_ema_net1/ema_net1_final.pth',
        'aux': '/data/output-ai/shruthi.gowda/continual/clmm_derpp_v2n1/results/class-il/seq-cifar10/derpp_mm_eman1_model/derppema-n1m-200-seq-cifar10-kl1.0-0.1-up0.2-s2/net_final.pth'
    },
    'seq-cifar100':{
        'der': '/data/output-ai/shruthi.gowda/continual/baseline/results/class-il/seq-cifar100/derpp/cll-derpp-200-seq-cifar100-s0/net_final.pth',
        'er': '/data/output-ai/shruthi.gowda/continual/baseline/results/class-il/seq-cifar100/er/cll-er-200-seq-cifar100-s0/net_final.pth',
        'cls-er': '/data/output-ai/fahad.sarfraz/baseline/task_models/seq-cifar100/c100-5-200-param-v1-s-0/task_5_stable_model.ph',
        'aux_ema': '/data/output-ai/shruthi.gowda/continual/clmm_derpp_v2n1_l2/results/class-il/seq-cifar100/derpp_mm_eman1/cll-mod-200-seq-cifar100-l20.10.01-up0.08-s1_ema_net1/ema_net1_4.pth'
    }
}

lst_buffer_size = [200]
device = 'cuda'
severity = [1,2,3,4,5]

results = {
    'id': [],
    'method': [],
    'corruption': [],
    'severity': [],
    'accuracy': [],
}

for buffer_size in lst_buffer_size:
    # buffer_size = 500
    print('=' * 50)
    print(f'Buffer Size = {buffer_size}')
    print('=' * 50)

    for method in lst_methods[dataset_name]:

        model_path = lst_methods[dataset_name][method]
        model = torch.load(model_path).to(device)

        for corrupt_name in corrupt_list:
            for sev in severity:

                args = arg_class(corrupt_name, sev)
                transf, transf_aux = transform_test(args)
                if dataset_name == 'seq-cifar10':
                    if 'aux' in method:
                        dataset = CIFAR10('data', train=False, download=True, transform=transf)
                    else:
                        dataset = CIFAR10('data', train=False, download=True, transform=transf_aux)
                else:
                    if 'aux' in method:
                        dataset = CIFAR100('data', train=False, download=True, transform=transf)
                    else:
                        dataset = CIFAR100('data', train=False, download=True, transform=transf_aux)

                data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
                accs = evaluate(model, data_loader, device=device)

                print('Accuracy:', accs)
                results['method'].append(method)
                results['id'].append(model_path)
                results['corruption'].append(corrupt_name)
                results['severity'].append(sev)
                results['accuracy'].append(accs)

            df = pd.DataFrame(results)
            df.to_csv('/volumes2/continual_learning/paper/analysis/corrupt/cor_cif100_all.csv')

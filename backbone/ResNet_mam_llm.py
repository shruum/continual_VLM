# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, relu

from backbone import MammothBackbone
from backbone.utils.llama import LLaMATransformer


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.

    Args:
        in_planes: number of input channels
        out_planes: number of output channels
        stride: stride of the convolution

    Returns:
        convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.

        Args:
            in_planes: the number of input channels
            planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.return_prerelu = False
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, input_size)

        Returns:
            output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        if self.return_prerelu:
            self.prerelu = out.clone()

        out = relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)

        out = self.relu(out)



        return out

class ResNetMamLLM(MammothBackbone):
    """
    ResNet network architecture. Designed for complex norm_datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, llm_block: str,
                 llama_config={"dim": 4096, "multiple_of": 1024,
                               "n_heads": 32, "n_layers": 32, "norm_eps": 1.0e-5,
                               "vocab_size": -1, "first_layer": 31, "n_kv_heads": 8,
                               "ffn_dim_multiplier": 1.3}) -> None:
        """
        Instantiates the layers of the network.

        Args:
            block: the basic ResNet block
            num_blocks: the number of blocks per layer
            num_classes: the number of output classes
            nf: the number of filters
        """
        super(ResNetMamLLM, self).__init__()
        self.return_prerelu = False
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        if self.num_classes == 2:
            self.embed_dim = 73728
        elif self.num_classes == 200:
            self.embed_dim = 32768
        else:
            self.embed_dim = 8192
        self.feature_dim = nf * 8 * block.expansion
        self.llm_block =llm_block
        # self.llm = LLaMATransformer(llama_config).cuda()
        # self.llm_dim_mapper1 = nn.Linear(self.embed_dim, 4096, bias=False)
        # self.llm_dim_mapper1 = nn.Linear(4096, self.embed_dim, bias=False)

        if self.llm_block == 'clip':
            from transformers import CLIPModel
            self.llm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            llm_hidden_size = self.llm.config.text_config.hidden_size
            print("Loading CLIP LLM model")
        elif self.llm_block == 'sent_transf':
            from sentence_transformers import SentenceTransformer
            self.llm = SentenceTransformer('all-MiniLM-L6-v2')
            llm_hidden_size = self.llm.get_sentence_embedding_dimension()
            print("Loading Sent Transformer LLM model")

        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm_dim_mapper1 = nn.Linear(self.embed_dim, llm_hidden_size)
        self.llm_dim_mapper2 = nn.Linear(llm_hidden_size, self.feature_dim)

        self.classifier = nn.Linear(nf * 8 * block.expansion, num_classes)


    def set_return_prerelu(self, enable=True):
        self.return_prerelu = enable
        for c in self.modules():
            if isinstance(c, self.block):
                c.return_prerelu = enable

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.

        Args:
            block: ResNet basic block
            planes: channels across the network
            num_blocks: number of blocks
            stride: stride

        Returns:
            ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, *input_shape)
            returnt: return type (a string among 'out', 'features', 'both', and 'full')

        Returns:
            output tensor (output_classes)
        """
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512
        if returnt == 'features':
            return feature

        out_4_flat = out_4.view(out_4.size(0), -1)  # (batch_size, 8192)
        out_4_proj = self.llm_dim_mapper1(out_4_flat)  # Flatten
        if self.llm_block == 'clip':
            llm_output = self.llm.text_model.encoder(inputs_embeds=out_4_proj.unsqueeze(1)).last_hidden_state
        elif self.llm_block == 'sent_transf':
            transformer_model = self.llm[0].auto_model  # Get the Hugging Face transformer model
            transformer_output = transformer_model(
                inputs_embeds=out_4_proj.unsqueeze(1),  # Add the sequence dimension
                output_hidden_states=False,  # Don't need all hidden states, just the last output
                return_dict=True  # Use Hugging Face's return dict
            )
            llm_output = transformer_output.last_hidden_state

        feature_l = self.llm_dim_mapper2(llm_output.squeeze(1))

        # feature_l = out_4.view(out_4.shape[0], self.embed_dim, -1).permute(0, 2, 1)
        # feature_l = self.llm_dim_mapper1(feature_l)
        # feature_l = self.llm(feature_l)
        # feature_l = self.llm_dim_mapper2(feature_l) #[32, 16, 512]
        # feature_l = torch.mean(feature_l, dim=1)  # feature_l will now have shape [32, 512]
        out = self.classifier(feature_l)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feature)
        # elif returnt == 'llm_all':
        #     return (out, feature, feature_l)
        elif returnt == 'full':
            return out, [
                out_0 if not self.return_prerelu else out_0_t,
                out_1 if not self.return_prerelu else self.layer1[-1].prerelu,
                out_2 if not self.return_prerelu else self.layer2[-1].prerelu,
                out_3 if not self.return_prerelu else self.layer3[-1].prerelu,
                out_4 if not self.return_prerelu else self.layer4[-1].prerelu
            ]

        raise NotImplementedError("Unknown return type. Must be in ['out', 'features', 'both', 'full'] but got {}".format(returnt))


def resnet18mamllm(nclasses: int, nf: int = 64, llm_block='clip') -> ResNetMamLLM:
    """
    Instantiates a ResNet18 network.

    Args:
        nclasses: number of output classes
        nf: number of filters

    Returns:
        ResNet network
    """
    return ResNetMamLLM(BasicBlock, [2, 2, 2, 2], nclasses, nf, llm_block)


def resnet34mamllm(nclasses: int, nf: int = 64) -> ResNetMamLLM:
    """
    Instantiates a ResNet34 network.

    Args:
        nclasses: number of output classes
        nf: number of filters

    Returns:
        ResNet network
    """
    return ResNetMamLLM(BasicBlock, [3, 4, 6, 3], nclasses, nf)

def resnet50mamllm(nclasses: int, nf: int=64):
    return ResNetMamLLM(Bottleneck, [3,4,6,3], nclasses, nf)
import torch
import torch.nn as nn
from torch import optim
import sys
sys.path.append("..")
from .frozen.model import OPTCaptioningModel
from models.utils.continual_model import ContinualModel
from utils.args import *
from utils.buffer import Buffer
from utils.aux_utils import AuxiliaryNet
from utils.vision_lang import loss_vlm

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Vision and language Continual learning via Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_auxiliary_args(parser)

    return parser


class VQAER(ContinualModel):
    NAME = 'vl_er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual', 'multi-modal']

    def __init__(self, backbone, loss, args, transform, config=dict()):
        super(VQAER, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.aux = AuxiliaryNet(self.args, self.device)
        # self.addit_models = ['net']
        
        self.config = config
        self.model = OPTCaptioningModel(backbone, config)
        self.loss_fn = nn.CrossEntropyLoss()

        self.num_image_tokens = 2
        self.task = 0
        self.iteration = 0

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    def check_model(self, **kwargs):
        with torch.no_grad():
            self.model.eval()
            out = self.model.generate(**kwargs)
        return out

    def observe(self, inputs, labels, not_aug_inputs, class_names=None, input_ids=None, attention_mask=None, image_token_mask=None):

        real_batch_size = inputs.shape[0]
        loss = 0
        loss_dict = {}
        self.opt.zero_grad()

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.model.image_encoder(inputs)
        print(outputs[0])
        features = self.model.image_encoder.forward_features(inputs)
        loss_ce1 = self.loss(outputs[0], labels)

        V = self.model.text_encoder.config.vocab_size
        N = self.num_image_tokens

        kwargs = {
            'pixel_values': inputs,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image_token_mask': image_token_mask,
        }

        output = self.forward(**kwargs)
        labels = input_ids.clone()
        shift_logits = output.logits[..., N:-1, :].contiguous()
        shift_labels = labels[..., N+1:].contiguous()
        
        loss_aux = self.loss_fn(shift_logits.view(-1, V), shift_labels.view(-1))
        
        loss = loss_ce1 + loss_aux

        self.aux.collate_loss(loss_dict, loss_ce=loss_ce1, loss_aux=loss_aux, m1=True)
        if hasattr(self, 'writer'):
            for loss_name, loss_item in loss_dict.items():
                self.writer.add_scalar('Task {}/{}'.format(self.task, loss_name), loss_item,
                                          global_step=self.iteration)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset):
        print('Saving Model')
        self.task += 1
        self.save_models(dataset)


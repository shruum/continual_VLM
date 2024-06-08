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

IMAGE_TOKEN = "<image>"
NUM_IMAGE_TOKENS = 2
NUM_TEXT_TOKENS = 6
SPECIAL_TOKEN_DICT = {'additional_special_tokens': [IMAGE_TOKEN]}

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Vision and language Continual learning via Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_auxiliary_args(parser)

    return parser

def get_tokens(model, inputs):
    I = model.model.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    input_ids = [I for i in range(NUM_IMAGE_TOKENS)] + inputs['input_ids']
    attention_mask = [1 for i in range(NUM_IMAGE_TOKENS)] + inputs['attention_mask']
    return {
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor(attention_mask),
    }


class VQAER(ContinualModel):
    NAME = 'vqa_er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual', 'multi-modal']

    def __init__(self, backbone, loss, args, transform, config=dict()):
        super(VQAER, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.aux = AuxiliaryNet(self.args, self.device)
        self.addit_models = ['net']
        
        self.config = config
        self.model = OPTCaptioningModel(backbone, config)
        self.loss_fn = nn.CrossEntropyLoss()

        self.num_image_tokens = NUM_IMAGE_TOKENS
        self.task = 0
        self.iteration = 0
        
        if not IMAGE_TOKEN in self.model.tokenizer.all_special_tokens:
            self.model.tokenizer.add_special_tokens(SPECIAL_TOKEN_DICT)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def get_tokens(self, inputs):
        I = self.model.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        input_ids = [[I for i in range(NUM_IMAGE_TOKENS)] + inp for inp in inputs['input_ids']]
        attention_mask = [[1 for i in range(NUM_IMAGE_TOKENS)] + inp for inp in inputs['attention_mask']]
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
        }
    
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
        # print(outputs[0])
        #features = self.model.image_encoder.forward_features(inputs)
        # print(outputs.shape, labels.shape)
        loss_ce1 = self.loss(outputs, labels)
        
        
        tok_labels = self.model.tokenizer([f"This is an image of {class_names[label.item()]}." for label in labels], padding=True)
        batch = self.get_tokens(tok_labels)
        image_token_mask = batch['input_ids'] == self.model.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        image_token_mask = image_token_mask[0]
        # image_token_mask[2:] = False

        # kwargs = {
        #     'pixel_values': inputs,
        #     'input_ids': batch['input_ids'],
        #     'attention_mask': batch['attention_mask'],
        #     'image_token_mask': image_token_mask,
        # }

        V = self.model.text_encoder.config.vocab_size
        N = self.num_image_tokens + NUM_TEXT_TOKENS
        
        # image_token_mask = batch.input_ids == self.model.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        
        
        # print(batch['input_ids'])

        kwargs = {
            'pixel_values': inputs,
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
            'image_token_mask': image_token_mask.to(self.device),
        }

        output = self.forward(**kwargs)
        tok_labels = batch['input_ids'].clone().to(self.device)
        shift_logits = output.logits[..., N:-1, :].contiguous()
        shift_labels = tok_labels[..., N+1:].contiguous()
        
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
        self.save_models(dataset, self.model.image_encoder)


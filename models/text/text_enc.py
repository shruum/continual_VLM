import json
import clip
import torch
import random
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from models.text.text_util import CLIPTEXT
from torch.nn import init
from transformers import AutoTokenizer, AutoModel

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to get text embeddings
GPT_PATH_CIFAR = "cl_datasets/metadata/lvis_gpt3_text-davinci-002_descriptions_author.json"
GPT_PATH = "/volumes1/datasets/tinyimagenet_description.json"

# Define the class TextEncoder
class TextEncoder():
    def __init__(self, encoder_type, device, pretrain=True):
        self.pretrain = pretrain
        self.device = device
        self.encoder_type = encoder_type
        self.text_encoder = self.build_text_encoder(pretrain)

    def build_text_encoder(self, pretrain=True):
        if self.encoder_type == "clip":
            return self.build_text_clip_encoder(pretrain)
        elif self.encoder_type == "sent_transf":
            return self.build_text_minlm_encoder()
        elif self.encoder_type == "sent_transf_large":
            return self.build_text_roberta_encoder()
        elif self.encoder_type == "bert":
            return self.build_text_bert_encoder()
        elif self.encoder_type == "code_lm":
            return self.build_text_codelm_encoder()
        else:
            raise ValueError('Please define a valid text encoder type: "clip", "sent_transf", or "bert"')

    # Function to build BERT encoder
    def build_text_bert_encoder(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        return tokenizer, model

    # Function to encode sentences using BERT
    def encode_with_bert(self, sentences):
        tokenizer, model = self.text_encoder
        model.eval()
        inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Use CLS token representation
        return embeddings

    def build_text_clip_encoder(self, pretrain=True):
        text_encoder = CLIPTEXT().to(self.device)
        text_encoder.eval()
        if pretrain == "True":
            pretrained_model, _ = clip.load("ViT-B/32", device=self.device) #RN50
            state_dict = pretrained_model.state_dict()
            to_delete_keys = ["logit_scale", "input_resolution", "context_length", "vocab_size"] + \
                             [k for k in state_dict.keys() if k.startswith('visual.')]
            for k in to_delete_keys:
                if k in state_dict:
                    del state_dict[k]
            print('Loading pretrained CLIP')
            text_encoder.load_state_dict(state_dict)
        return text_encoder

    def build_text_minlm_encoder(self):
        model_name = 'all-MiniLM-L6-v2'
        if self.pretrain == "False":
            text_encoder = SentenceTransformer(model_name).to(self.device)
            print("Initializing LLM with Random weights")
            for name, param in text_encoder.named_parameters():
                # if param.requires_grad:
                #     if param.dim() >= 2:  # Check if tensor has 2 or more dimensions
                #         init.xavier_uniform_(param)
                #     else:
                #         init.zeros_(param)
                if param.dim() >= 2:  # For multi-dimensional tensors (e.g., weight matrices)
                    with torch.no_grad():
                        param.copy_(param.view(-1)[torch.randperm(param.numel())].view(param.size()))
                elif param.dim() == 1:  # For one-dimensional tensors (e.g., biases)
                    with torch.no_grad():
                        param.copy_(param[torch.randperm(param.size(0))])
        else:
            text_encoder = SentenceTransformer(model_name).to(self.device)
        text_encoder.eval()
        return text_encoder

    def build_text_roberta_encoder(self):
        model_name = 'all-distilroberta-v1' #all-roberta-large-v1'
        text_encoder = SentenceTransformer(model_name).to(self.device)
        text_encoder.eval()
        return text_encoder
    def build_text_codelm_encoder(self):
        model_name = "microsoft/codebert-base"  # Replace with your desired CodeLM model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)

        return tokenizer, model

    def encode_with_codelm(self, sentences):
        tokenizer, model = self.text_encoder
        model.eval()
        inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token representation
        return embeddings

    def encode_sentences(self, sentences):
        if self.encoder_type == "clip":
            sentences = torch.cat([self.text_encoder.tokenize(sent) for sent in sentences]).to(self.device)
            with torch.no_grad():
                embeddings = self.text_encoder.encode_text(sentences).detach()
        elif self.encoder_type == "sent_transf" or self.encoder_type == "sent_transf_large":
            embeddings = self.text_encoder.encode(sentences, convert_to_tensor=True, device=self.device).detach()
        elif self.encoder_type == "bert":
            embeddings = self.encode_with_bert(sentences).detach()
        elif self.encoder_type == "code_lm":
            embeddings = self.encode_with_codelm(sentences).detach()
        else:
            raise ValueError('Please define a valid text encoder type: "clip", "sent_transf", or "bert"')
        return embeddings


# Function to get text embeddings using TextEncoder class
def get_text_embeddings(model, labels, dataset=None, dataloader=None, prompt='a '):

    text_encoder = model.text_encoder
    with open(model.args.gpt_path, 'r') as f:
        descriptions = json.load(f)

    sentences = []
    for label in labels:
        if model.args.dataset == 'seq-tinyimg':
            task = dataset.i
            cls_idx = dataset.img_paths[task]['class_to_idx']
            class_name = list(cls_idx.keys())[list(cls_idx.values()).index(label)]
            sentences.append(descriptions[class_name])
        elif model.args.dataset == 'dn4il':
            class_name = str(label.item())
            sentences.append(descriptions[class_name])
        elif model.args.dataset == 'seq-cifar10' or model.args.dataset == 'seq-cifar10-vit' or model.args.dataset == 'cifartint':
            class_name = dataset.CLASS_ID[label.item()]
            sentences.append(descriptions[class_name][0])
        elif model.args.dataset == 'seq-cifar100':
            class_id = str(label.item())
            sentences.append(descriptions[class_id][0])
        elif model.args.dataset == 'cifar10':
            class_name = dataset.CLASS_ID[label.item()]
            sentences.append(descriptions[class_name][0])
        elif model.args.dataset == 'cifar100':
            class_id = str(label.item())
            sentences.append(descriptions[class_id][0])
        elif model.args.dataset == 'cifar10_imb':
            class_name = dataset.CLASS_ID[label.item()]
            sentences.append(descriptions[class_name][0])
        elif model.args.dataset == 'celeba':
            class_name = dataset.CLASS_ID[label.item()]
            if len(descriptions[class_name]) > 1:
                ind = random.randint(0,4)
            else:
                ind = 0
            sentences.append(descriptions[class_name][ind])
        elif model.args.dataset == 'waterbirds':
            class_name = dataset.CLASS_ID[label.item()]
            if len(descriptions[class_name]) > 1:
                ind = random.randint(0,2)
            else:
                ind = 0
            sentences.append(descriptions[class_name][ind])
        elif model.args.dataset == 'tinyimagenet':
            CLASS_ID = dataloader.dataset.class_to_idx
            class_name = list(CLASS_ID.keys())[label.item()]
            sentences.append(descriptions[class_name])
        elif model.args.dataset == 'imagenet100' or model.args.dataset == 'imagenet1k':
            CLASS_ID = dataloader.dataset.class_to_idx
            class_name = list(CLASS_ID.keys())[label.item()]
            sentences.append(descriptions[class_name])
        else:
            sentences.append(descriptions[label])

    embeddings = text_encoder.encode_sentences(sentences)
    return embeddings

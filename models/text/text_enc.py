import json
import clip
import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from models.text.text_util import CLIPTEXT

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to get text embeddings
GPT_PATH_CIFAR = "cl_datasets/metadata/lvis_gpt3_text-davinci-002_descriptions_author.json"
GPT_PATH = "/volumes1/datasets/tinyimagenet_description.json"

# Define the class TextEncoder
class TextEncoder():
    def __init__(self, encoder_type, device, pretrain=True):
        self.device = device
        self.encoder_type = encoder_type
        self.text_encoder = self.build_text_encoder(pretrain)

    def build_text_encoder(self, pretrain=True):
        if self.encoder_type == "clip":
            return self.build_text_clip_encoder(pretrain)
        elif self.encoder_type == "sent_transf":
            return self.build_text_minlm_encoder()
        elif self.encoder_type == "bert":
            return self.build_text_bert_encoder()
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
        if pretrain:
            pretrained_model, _ = clip.load("ViT-B/32", device=self.device)
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
        text_encoder = SentenceTransformer(model_name).to(self.device)
        text_encoder.eval()
        return text_encoder

    def encode_sentences(self, sentences):
        if self.encoder_type == "clip":
            sentences = torch.cat([self.text_encoder.tokenize(sent) for sent in sentences]).to(self.device)
            with torch.no_grad():
                embeddings = self.text_encoder.encode_text(sentences).detach()
        elif self.encoder_type == "sent_transf":
            embeddings = self.text_encoder.encode(sentences, convert_to_tensor=True, device=self.device).detach()
        elif self.encoder_type == "bert":
            embeddings = self.encode_with_bert(sentences).detach()
        else:
            raise ValueError('Please define a valid text encoder type: "clip", "sent_transf", or "bert"')
        return embeddings


# Function to get text embeddings using TextEncoder class
def get_text_embeddings(text_encoder, labels, dataset=None, prompt='a '):
    with open(dataset.args.gpt_path, 'r') as f:
        descriptions = json.load(f)

    task = dataset.i
    sentences = []
    for label in labels:
        if dataset.args.dataset == 'seq-tinyimg':
            cls_idx = dataset.img_paths[task]['class_to_idx']
            class_name = list(cls_idx.keys())[list(cls_idx.values()).index(label)]
            sentences.append(descriptions[class_name])
        elif dataset.args.dataset == 'dn4il':
            class_name = str(label.item())
            sentences.append(descriptions[class_name])
        elif dataset.args.dataset == 'seq-cifar10':
            class_name = dataset.CLASS_ID[label.item()]
            sentences.append(descriptions[class_name][0])
        else:
            sentences.append(descriptions[label])

    embeddings = text_encoder.encode_sentences(sentences)
    return embeddings

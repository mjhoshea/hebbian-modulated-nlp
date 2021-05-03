import math
import random
from collections import defaultdict

import torch

from torch import nn
from torch.autograd import Variable
from torch.nn import init, Parameter
from torch.nn import functional as F
from transformers import AlbertModel, AlbertTokenizer, BertTokenizer, BertModel

from src.models.anml.plastic_layers import PlasticModulated, Plastic


class TransformerClsModel(nn.Module):

    def __init__(self, model_name, n_classes, max_length, out_layer, device, modulation):
        super(TransformerClsModel, self).__init__()
        self.n_classes = n_classes
        self.max_length = max_length
        self.device = device
        self.modulation = modulation
        if model_name == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')
        elif model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        else:
            raise NotImplementedError

        if out_layer == 'plastic':
            self.out_layer = Plastic(768, n_classes)
        elif out_layer == 'plastic_modulated':
            self.out_layer = PlasticModulated(768, n_classes)
        else:
            self.out_layer = nn.Linear(768, n_classes)

        self.to(self.device)

    def encode_text(self, text):
        encode_result = self.tokenizer.batch_encode_plus(text, return_token_type_ids=False, max_length=self.max_length,
                                                         truncation=True, padding='max_length', return_tensors='pt')
        for key in encode_result:
            encode_result[key] = encode_result[key].to(self.device)
        return encode_result

    def forward(self, inputs, modulation, out_from='full', is_training=True):
        if out_from == 'full':
            _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            if self.modulation == 'input':
                out = modulation * out
                out = self.out_layer(out, is_training)
            elif self.modulation == 'hebbian':
                out = self.out_layer(out, modulation, is_training)
            elif self.modulation == 'double':
                mod1 = modulation[:, :768]
                mod2 = modulation[:, 768:]
                out = mod1 * out
                out = self.out_layer(out, mod2, is_training)
            else:
                out = self.out_layer(out, is_training)


        elif out_from == 'transformers':
            _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        elif out_from == 'linear':
            out = self.out_layer(inputs, modulation, is_training)
        else:
            raise ValueError('Invalid value of argument')
        return out

    def vis(self, inputs, modulation, out_from='full', is_training=True):
        pre_mod = None
        post_mod = None
        file_name = None

        if out_from == 'full':
            _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            if self.modulation == 'input':
                pre_mod = out.copy()

                out = modulation * out
                out = self.out_layer(out, is_training)

                post_mod = out.copy()
                file_name = 'input_modulation'

            elif self.modulation == 'hebbian':
                pre_mod = out.copy()

                out = self.out_layer(out, modulation, is_training)
                post_mod = out.copy()

                file_name = 'hebbian_modulation'
            elif self.modulation == 'double':
                mod1 = modulation[:, :768]
                mod2 = modulation[:, 768:]

                pre_mod = out

                out = mod1 * out
                a = out.copy()
                out = self.out_layer(out, mod2, is_training)
                b = out.copy()
                post_mod = torch.array([a, b])

                file_name = "double_modulation"
            else:
                out = self.out_layer(out, is_training)

        return pre_mod, modulation, post_mod, file_name




class TransformerRLN(nn.Module):

    def __init__(self, model_name, max_length, device):
        super(TransformerRLN, self).__init__()
        self.max_length = max_length
        self.device = device
        if model_name == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')
        elif model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        else:
            raise NotImplementedError
        self.to(self.device)

    def encode_text(self, text):
        encode_result = self.tokenizer.batch_encode_plus(text, return_token_type_ids=False, max_length=self.max_length,
                                                         truncation=True, padding='max_length', return_tensors='pt')
        for key in encode_result:
            encode_result[key] = encode_result[key].to(self.device)
        return encode_result

    def forward(self, inputs):
        _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return out

class LinearPLN(nn.Module):

    def __init__(self, in_dim, out_dim, device):
        super(LinearPLN, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear.to(device)

    def forward(self, input):
        out = self.linear(input)
        return out


class TransformerNeuromodulator(nn.Module):

    def __init__(self, n_classes, model_name, device, modulation):
        if modulation == 'input':
            self.output_dim = 768
        elif modulation == 'hebbian':
            self.output_dim = 768*n_classes
        else:
            self.output_dim = 768*(n_classes + 1)

        super(TransformerNeuromodulator, self).__init__()
        self.device = device
        self.n_classes = n_classes
        if model_name == 'albert':
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')
        elif model_name == 'bert':
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        else:
            raise NotImplementedError
        self.encoder.requires_grad = False
        self.linear = nn.Sequential(nn.Linear(768, 768),
                                    nn.ReLU(),
                                    nn.Linear(768, self.output_dim),
                                    nn.Sigmoid())
        self.to(self.device)

    def forward(self, inputs, out_from='full'):
        _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        out = self.linear(out)
        return out


class ReplayMemory:

    def __init__(self, write_prob, tuple_size):
        self.buffer = []
        self.write_prob = write_prob
        self.tuple_size = tuple_size

    def write(self, input_tuple):
        if random.random() < self.write_prob:
            self.buffer.append(input_tuple)

    def read(self):
        return random.choice(self.buffer)

    def write_batch(self, *elements):
        element_list = []
        for e in elements:
            if isinstance(e, torch.Tensor):
                element_list.append(e.tolist())
            else:
                element_list.append(e)
        for write_tuple in zip(*element_list):
            self.write(write_tuple)

    def read_batch(self, batch_size):
        contents = [[] for _ in range(self.tuple_size)]
        for _ in range(batch_size):
            read_tuple = self.read()
            for i in range(len(read_tuple)):
                contents[i].append(read_tuple[i])
        return tuple(contents)

    def len(self):
        return len(self.buffer)

    def reset_memory(self):
        self.buffer = []

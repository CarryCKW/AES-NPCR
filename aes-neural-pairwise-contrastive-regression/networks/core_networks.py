# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from transformers import BertModel

# file type contains {'bert-base-uncased', 'roberta-base', 'xlnet-base-cased'}
# should keep pace with file in ./../reader.py
file = 'bert-base-uncased'
bert_model = BertModel.from_pretrained(file)
bert_model.cuda()


class mlp(nn.Module):
    def __init__(self, in_f, out_f):
        super(mlp, self).__init__()
        self.layer1 = nn.Linear(in_f, 768)
        self.active = nn.Tanh()
        self.layer2 = nn.Linear(768, out_f)

    def forward(self, x):
        out = self.layer1(x)

        return out


class npcr_model(nn.Module):
    def __init__(self, maxSq=512):
        super(npcr_model, self).__init__()

        self.embedding = bert_model
        self.dropout = nn.Dropout(0.5)

        self.nn1 = nn.Linear(768, 768)
        self.output = nn.Linear(768, 1, bias=False)

        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias_ih' in name or 'bias_hh' in name)
        # nn.init.uniform(self.embed.weight.data, a=-0.5, b=0.5)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, x0, x1):
        x0_embed = self.embedding(x0)[1]
        x1_embed = self.embedding(x1)[1]

        # the linear layer nn1 can be replaced by MLP(the above or overwite by yourself)
        x0_nn1 = self.nn1(x0_embed)
        x1_nn1 = self.nn1(x1_embed)

        x0_nn1_d = self.dropout(x0_nn1)
        x1_nn1_d = self.dropout(x1_nn1)

        diff_x = (x0_nn1_d - x1_nn1_d)
        y = self.output(diff_x)

        y = torch.sigmoid(y)

        return y




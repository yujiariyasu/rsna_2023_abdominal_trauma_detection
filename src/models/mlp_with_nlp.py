import torch
import transformers
from transformers import AutoModel, AdamW, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from pdb import set_trace as st

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine.float() > self.th, phi, cosine.float() - self.mm)
        if label is not None:
            one_hot = torch.zeros(cosine.size(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            output = cosine
        output *= self.s

        return output

class Mlp_with_nlp_arcface(nn.Module):
    def __init__(self, model_name, num_classes, embedding_size, s=30.0, m=0.5, ls_eps=0.0):
        super(Mlp_with_nlp_arcface, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.text_model = nn.Sequential(
            nn.BatchNorm1d(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        self.num_vals_model = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Linear(2, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        # self.neck = nn.Sequential(
        #     nn.BatchNorm1d(512),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(p=0.2),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.Dropout(p=0.2),
        #     nn.ReLU(),

            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.Dropout(p=0.2),
            # nn.ReLU(),
            # nn.Linear(128, embedding_size),
            # nn.BatchNorm1d(embedding_size),
            # nn.Dropout(p=0.2),

            # nn.Linear(256, embedding_size),
            # nn.BatchNorm1d(embedding_size),
            # nn.Dropout(p=0.2),
        # )

        self.neck = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(256, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.Dropout(p=0.2),
        )

        self.ln = nn.LayerNorm(embedding_size)

        self.fc = ArcMarginProduct(embedding_size,
                                   num_classes,
                                   s=s,
                                   m=m,
                                   easy_margin=False)

    def forward(self, ids, mask, num_vals, labels=None):
        ids, mask = ids.squeeze(1), mask.squeeze(1)
        text_out = self.bert(input_ids=ids, attention_mask=mask)[1]
        text_out = self.text_model(text_out)
        num_vals = self.num_vals_model(num_vals)
        embedding = torch.cat([text_out, num_vals], axis=1)
        embedding = self.neck(embedding)
        embedding = self.ln(embedding)
        output = self.fc(embedding, labels)
        return output
    
    def extract(self, ids, mask, num_vals):
        ids, mask = ids.squeeze(1), mask.squeeze(1)
        text_out = self.bert(input_ids=ids, attention_mask=mask)[1]
        text_out = self.text_model(text_out)
        num_vals = self.num_vals_model(num_vals)
        embedding = torch.cat([text_out, num_vals], axis=1)
        embedding = self.neck(embedding)
        embedding = self.ln(embedding)
        return embedding

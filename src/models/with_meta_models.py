import torch
import torch.nn as nn
import timm
from pdb import set_trace as st
import torch.nn.functional as F
import math

class WithMetaModel(nn.Module):
    def __init__(self, base_model, model_name, pretrained=True, num_classes=1, meta_cols_in_features=6):
        super(WithMetaModel, self).__init__()

        meta_cols_out_features = 64
        self.model = base_model
        if 'swin' in model_name:
            in_features = self.model.head.in_features
            self.model.head = nn.Identity()
            self.fc = nn.Linear(in_features+meta_cols_out_features, num_classes)
        elif ('resnet' in model_name) | ('resnext' in model_name):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
            self.fc = nn.Linear(in_features+meta_cols_out_features, num_classes)
        elif 'efficientnet' in model_name:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
            self.fc = nn.Linear(in_features+meta_cols_out_features, num_classes)
        # elif 'vit' in model_name:
        #     in_features = self.model.head.in_features
        #     sys.path.append('/home/acc12347av/ml_pipeline/src')
        #     import timm055
        #     self.model = timm055.create_model(model_name, pretrained=pretrained, num_classes=1)
        #     self.model.avgpool = nn.Identity()
        #     self.model.head = nn.Identity()
        #     self.pooling = nn.Identity()
        elif 'beit' in model_name:
            in_features = self.model.head.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.head = nn.Identity()
            self.model.fc_norm = nn.Identity()
            self.pooling = nn.Identity()
            self.fc = nn.Linear(in_features+meta_cols_out_features, num_classes)
        elif 'convnext' in model_name:
            in_features = self.model.head.fc.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.head.fc = nn.Identity()
            self.fc = nn.Linear(in_features+meta_cols_out_features, num_classes)
        else:
            raise

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(meta_cols_in_features),
            nn.Linear(meta_cols_in_features, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.3),
        )

    def forward(self, input_):
        x, meta = input_
        # st()
        x = self.model(x)
        meta = self.mlp(meta) # bs, 64
        x = torch.cat([x, meta], 1)
        return self.fc(x)

    def extract_features(self, input_):
        x, meta = input_
        x = self.model(x)
        meta = self.mlp(meta) # bs, 64
        return torch.cat([x, meta], 1)

    def extract_with_features(self, input_):
        x, meta = input_
        x = self.model(x)
        meta = self.mlp(meta) # bs, 32
        x = torch.cat([x, meta], 1)
        return self.fc(x), x

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as st

class MultiLayerPerceptron(nn.Module):
    def __init__(self, num_classes, input_num, dropout=0.3):
        super(MultiLayerPerceptron, self).__init__()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_num),
            nn.Linear(input_num, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, labels=None):
        x = self.mlp(x)
        return x

class MultiLayerPerceptron3(nn.Module):
    def __init__(self, num_classes, input_num, dropout=0.3):
        super(MultiLayerPerceptron3, self).__init__()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_num),
            nn.Linear(input_num, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, labels=None):
        x = self.mlp(x)
        return x

class MultiLayerPerceptron5(nn.Module):
    def __init__(self, num_classes, input_num, dropout=0.3):
        super(MultiLayerPerceptron5, self).__init__()
        self.bowel = nn.Linear(9, 2)
        self.extravasation = nn.Linear(9, 2)
        self.kidney = nn.Linear(33, 3)
        self.liver = nn.Linear(27, 3)
        self.spleen = nn.Linear(27, 3)

    def forward(self, x, labels=None):
        bowel = self.bowel(x[:, :9])
        extravasation = self.extravasation(x[:, 9:18])
        kidney = self.kidney(x[:, 18:51])
        liver = self.liver(x[:, 51:78])
        spleen = self.spleen(x[:, 78:])
        return torch.cat([bowel, extravasation, kidney, liver, spleen], 1)

class MultiLayerPerceptron4(nn.Module):
    def __init__(self, num_classes, input_num, dropout=0.3):
        super(MultiLayerPerceptron4, self).__init__()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_num),
            nn.Linear(input_num, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, labels=None):
        x = self.mlp(x)
        return x

class MultiLayerPerceptron2(nn.Module):
    def __init__(self, num_classes, input_num, dropout=0.4):
        super(MultiLayerPerceptron2, self).__init__()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_num),
            nn.Linear(input_num, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(512, 1028),
            nn.BatchNorm1d(1028),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(1028, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, labels=None):
        x = self.mlp(x)
        return x

from pathlib import Path
from pprint import pprint
import timm
from src.utils.metric_learning_loss import *
from src.utils.metrics import *
from src.utils.loss import *
import os
import torch.nn as nn
import pandas as pd
import numpy as np
from pdb import set_trace as st
from sklearn.preprocessing import OneHotEncoder

from src.utils.augmentations.strong_aug import *
from src.utils.augmentations.augmentation import *
from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error, average_precision_score, accuracy_score
import segmentation_models_pytorch as smp
from src.models.segmentation_3d_models import convert_3d, TimmSegModel

def my_accuracy_score(true, pred):
    return accuracy_score(true, pred.argmax(1))

bce_loss = nn.BCEWithLogitsLoss()
lovasz_loss = smp.losses.LovaszLoss(mode='binary', per_image=False)
tversky_loss = smp.losses.TverskyLoss(mode='binary', log_loss=False, from_logits=True)

def bce_lovasz(output, target):
    return (0.5 * bce_loss(output, target)) + (0.5 * lovasz_loss(output, target))

def bce_lovasz_tversky_loss(output, target):
    bce_loss(output, target)
    lovasz_loss(output, target)
    tversky_loss(output, target)
    return (0.25 * bce_loss(output, target)) + (0.25 * lovasz_loss(output, target)) + (0.5 * tversky_loss(output, target))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

from scipy.spatial.distance import directed_hausdorff
import gc
def hd_dist(preds, targets):
    preds_coords = np.argwhere(preds) / np.array(preds.shape)
    targets_coords = np.argwhere(targets) / np.array(preds.shape)
    haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]
    return haussdorf_dist

def dice(im1, im2, empty_score=1.0):

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

class DiceMetric:
    def __call__(self, targets, logits):
        targets = targets.cpu().numpy()
        if targets.max()==0:
            return 0
        logits = sigmoid(logits.cpu().numpy())
        logits = np.round(logits)

        dices = []

        print('='*100)
        for class_id in range(logits.shape[1]):
            class_dices = []
            for i in range(0, logits.shape[0]):
                class_dices.append(dice(logits[i, class_id, :, :], targets[i, class_id, :, :]))

            class_dice_score = np.mean(class_dices)
            print('class_id/class_dice_score:', class_id, class_dice_score)
            dices.append(class_dice_score)
        dice_score = np.mean(dices)
        print('dice score mean:', dice_score)
        print('='*100)
        return dice_score

        print('\ndice:', vdice)

        # return (dice1 + dice2 + dice3) / 3.0
        n = 20
        vhd = 0
        chunk = len(logits)//n
        for i in range(n):
            # print(i)
            if i == n-1:
                lo = logits[i*chunk:]
                ta = targets[i*chunk:]
            else:
                lo = logits[i*chunk:(i+1)*chunk]
                ta = targets[i*chunk:(i+1)*chunk]

            h_dists1 = (1 - hd_dist(lo[:, 0, :, :], ta[:, 0, :, :])) * len(lo)
            h_dists2 = (1 - hd_dist(lo[:, 1, :, :], ta[:, 1, :, :])) * len(lo)
            h_dists3 = (1 - hd_dist(lo[:, 2, :, :], ta[:, 2, :, :])) * len(lo)
            vhd += (h_dists1 + h_dists2 + h_dists3) / 3.0
        vhd /= len(logits)
        print('Hausdorff:', vhd)
        score = (0.4 * vdice) + (0.6 * vhd)
        return score

JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss    = smp.losses.DiceLoss(mode='multilabel')
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

def criterion(y_pred, y_true):
    return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)

class Baseline:
    def __init__(self):
        self.gpu = 'small'
        self.compe = 'pharynx'
        self.batch_size = 32
        self.grad_accumulations = 1
        self.lr = 0.0001
        self.epochs = 130
        self.resume = False
        self.seed = 2022
        self.tta = 1
        self.predict_valid = False
        self.predict_test = False
        # self.criterion = criterion
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.image_size = (512, 512)
        self.transform = base_aug_v1
        self.num_classes = 6
        self.model_name = 'efficientnet-b1'
        self.model = smp.Unet(self.model_name, encoder_weights='imagenet', classes=self.num_classes, activation=None)
        self.metric = DiceMetric()
        # self.metric = None
        self.fp16 = True
        # self.optimizer = 'adam'
        self.optimizer = 'adamw'
        self.scheduler = 'WarmupCosineAnnealingLR'
        self.warmup_epochs = 1
        self.lr = 0.001
        self.eta_min = 5e-7
        self.t_max = 30
        self.train_by_all_data = False
        self.early_stop_patience = 8
        self.inference = False
        self.logit_to = None
        self.pretrained_path = None
        self.sync_batchnorm = False
        self.finetune_transform = base_aug_v1
        self.mixup = False
        self.arcface = False
        self.box_crop = None
        self.pad_square = False
        self.inference_only = False
        self.resume_epoch = 0
        self.save_preds = False
        self.save_targets = False
        self.save_top_k = 1
        self.seg_3d = False

################
def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def bce_dice(input, target, loss_weights=[1, 1]):
    loss1 = loss_weights[0] * torch.nn.BCEWithLogitsLoss()(input, target)
    loss2 = loss_weights[1] * dice_loss(input, target)
    return (loss1 + loss2) / sum(loss_weights)


class rsna_3d_seg_resnet18(Baseline):
    def __init__(self):
        super().__init__()
        self.compe = 'rsna'
        self.image_size = (128, 128, 128)
        self.num_classes = 5
        self.model_name = 'resnet18'
        self.model = TimmSegModel(self.model_name, pretrained=True, num_classes=self.num_classes)
        self.model = convert_3d(self.model)
        self.metric = None
        self.criterion = bce_dice
        self.batch_size = 4
        self.train_df_path = f'../input/train_for_3dseg.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.test_df_path = '../input/test_for_3dseg.csv'
        self.test_df = pd.read_csv(self.test_df_path)
        self.test_df['image_id'] = self.test_df.path.apply(lambda x: x.split('/')[-1].split('.')[0])
        self.seg_3d = True
        self.save_preds = True
        self.save_targets = True
        self.epochs = 300
        self.lr = 3e-3
        self.early_stop_patience = 300
        self.t_max = 300
        self.memo = '300 epochs'
        self.predict_valid = True
        self.predict_test = True

class rsna_3d_seg_resnet50(Baseline):
    def __init__(self):
        super().__init__()
        self.compe = 'rsna'
        self.image_size = (128, 128, 128)
        self.num_classes = 5
        self.model_name = 'resnet50'
        self.model = TimmSegModel(self.model_name, pretrained=True, num_classes=self.num_classes)
        self.model = convert_3d(self.model)
        self.metric = None
        self.criterion = bce_dice

        self.batch_size = 4
        self.train_df_path = '../input/train_for_3dseg.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.test_df_path = '../input/test_for_3dseg.csv'
        self.test_df = pd.read_csv(self.test_df_path)
        self.test_df['image_id'] = self.test_df.path.apply(lambda x: x.split('/')[-1].split('.')[0])
        self.seg_3d = True
        self.save_preds = True
        self.save_targets = False
        self.epochs = 300
        self.lr = 3e-3
        self.early_stop_patience = 300
        self.t_max = 300
        self.memo = '300 epochs'
        self.inference_only = True
        self.predict_valid = True
        self.predict_test = True
        self.gpu = 'v100'

from pathlib import Path
from pprint import pprint
import timm
from src.utils.metric_learning_loss import *
from src.utils.metrics import *
from src.utils.loss import *
from src.global_objectives import AUCPRLoss
import os
import torch.nn as nn
import pandas as pd
import numpy as np
from pdb import set_trace as st
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import pytorch_lightning as pl
from types import MethodType

from src.models.resnet3d_csn import *
from src.models.uniformerv2 import *
from src.models.rsna import *
from src.models.layers import AdaptiveConcatPool2d, Flatten
from src.models.ch_mdl_dolg_efficientnet import ChMdlDolgEfficientnet, ArcFaceLossAdaptiveMargin
from src.models.rsna_multi_image import MultiLevelModel2
from src.models.backbones import *
from src.models.group_norm import convert_groupnorm
from src.models.batch_renorm import convert_batchrenorm
from src.models.multi_instance import MultiInstanceModel, MetaMIL, AttentionMILModel, MultiInstanceModelWithWataruAttention
from src.models.resnet import resnet18, resnet34, resnet101, resnet152
from src.models.nextvit import NextVitNet
from src.models.model_4channels import get_attention, get_resnet34, get_attention_inceptionv3
from src.models.vae import VAE, ResNet_VAE
from src.models.model_with_arcface import ArcMarginProduct, AddMarginProduct, ArcMarginProductSubcenter, ArcMarginProductOutCosine, ArcMarginProductSubcenterOutCosine, PudaeArcNet, WithArcface, WhalePrev1stModel, Guie2
from src.models.with_meta_models import WithMetaModel

from src.utils.augmentations.strong_aug import *
from src.utils.augmentations.augmentation import *
from src.utils.augmentations.policy_transform import policy_transform
from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error, average_precision_score, accuracy_score

class Baseline:
    def __init__(self):
        self.compe = 'rsna'
        self.batch_size = 16
        self.grad_accumulations = 1
        self.lr = 0.0001
        self.epochs = 20
        self.resume = False
        self.seed = 2023
        self.tta = 1
        self.model_name = 'resnet50'
        self.num_classes = 1
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.transform = kuma_aug
        self.image_size = 384
        self.label_features = ['target']
        self.metric = roc_auc_score # AUC().torch # MultiAP().torch # MultiAUC().torch
        self.fp16 = True
        self.optimizer = 'adam'
        self.scheduler = 'CosineAnnealingWarmRestarts'
        self.eta_min = 5e-7
        self.train_by_all_data = False
        self.early_stop_patience = 1000
        self.inference = False
        self.predict_valid = True
        self.predict_test = False
        self.logit_to = None
        self.pretrained_path = None
        self.sync_batchnorm = False
        self.warmup_epochs = 1
        self.finetune_transform = base_aug_v1
        self.mixup = False
        self.arcface = False
        self.box_crop = None
        self.predicted_mask_crop = None
        self.pad_square = False
        self.resume_epoch = 0
        self.t_max=30
        self.save_top_k = 1
        self.meta_cols = []
        self.output_features = False
        self.force_use_model_path_config_when_inf = None
        self.reset_classifier_when_inf = False
        self.upsample = None
        self.in_chans = 3
        self.add_imsizes_when_inference = [(0, 0)]
        self.inf_fp16 = False
        self.distill = False
        self.reload_dataloaders_every_n_epochs = 0
        self.tranform_dataset_version = None
        self.no_trained_model_when_inf = False
        self.normalize_horiz_orientation = False
        self.upsample_batch_pos_n = None
        self.cut_200 = False
        self.affine_for_gbr = False
        self.half_dark = False
        self.crop_by_left_right_line_text = False
        self.use_wandb = None
        self.memo = ''
        self.use_last_ckpt_when_inference = False
        self.inference_only = False

class rsna_pretrain_maxvit(Baseline):
    def __init__(self):
        super().__init__()
        self.compe = 'rsna'
        self.model_name = 'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k'
        self.train_df_path = '../input/train_2classes_for_pretrain.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.test_df_path = '../input/train_2classes_for_pretrain.csv' # dummy
        self.test_df = pd.read_csv(self.test_df_path)
        self.predict_valid = False
        self.predict_test = False

        self.affine = False
        self.batch_size = 16
        self.grad_accumulations = 1
        self.lr = 1e-5
        self.image_size = 224
        self.transform = medical_v1
        self.cut_200 = False
        self.use_last_ckpt_when_inference = True
        self.epochs = 1
        self.memo = '2classes, stride2'
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=2)
        self.label_features = ['image_bowel_injury', 'image_extravasation_injury']
        self.metric = MultiAUC(label_features=self.label_features).torch
        self.strides = [2]
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')

class rsna_pretrain_convnext(Baseline):
    def __init__(self):
        super().__init__()
        self.compe = 'rsna'
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.train_df_path = '../input/train_2classes_for_pretrain.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.test_df_path = '../input/train_2classes_for_pretrain.csv' # dummy
        self.test_df = pd.read_csv(self.test_df_path)
        self.predict_valid = False
        self.predict_test = False

        self.affine = False
        self.batch_size = 16
        self.grad_accumulations = 1
        self.lr = 1e-5
        self.image_size = 384
        self.transform = medical_v1
        self.cut_200 = False
        self.use_last_ckpt_when_inference = True
        self.epochs = 1
        self.memo = '2classes, stride5, datav1'
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=2)
        self.label_features = ['image_bowel_injury', 'image_extravasation_injury']
        self.metric = MultiAUC(label_features=self.label_features).torch
        self.strides = [5]
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')

class rsna_pretrain_seresnext(Baseline):
    def __init__(self):
        super().__init__()
        self.compe = 'rsna'
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.train_df_path = '../input/train_2classes_for_pretrain.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.test_df_path = '../input/train_2classes_for_pretrain.csv' # dummy
        self.test_df = pd.read_csv(self.test_df_path)
        self.predict_valid = False
        self.predict_test = False

        self.affine = False
        self.batch_size = 16
        self.grad_accumulations = 1
        self.lr = 1e-4
        self.image_size = 384
        self.transform = medical_v1
        self.cut_200 = False
        self.use_last_ckpt_when_inference = True
        self.epochs = 1
        self.memo = '2classes, stride5, datav1'
        self.model = senet_mod(se_resnext50_32x4d, pretrained=True, num_classes=2)
        self.label_features = ['image_bowel_injury', 'image_extravasation_injury']
        self.metric = MultiAUC(label_features=self.label_features).torch
        self.strides = [2]
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')

class rsna_base(Baseline):
    def __init__(self):
        super().__init__()
        self.compe = 'rsna'
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.train_df_path = '../input/train2.5d_bad_fold.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 0]
        self.image_size = 256
        self.predict_valid = True
        self.predict_test = False
        self.affine = False
        self.transform = medical_v1
        self.cut_200 = False
        self.use_last_ckpt_when_inference = True
        self.multi_image = True
        self.memo = ''
        self.label_features = ['bowel_injury', 'extravasation_injury', 'kidney_healthy', 'kidney_low', 'kidney_high', 'liver_healthy', 'liver_low', 'liver_high', 'spleen_healthy', 'spleen_low', 'spleen_high']
        self.in_chans = 3
        self.lr = 1e-5
        self.batch_size = 8
        self.grad_accumulations = 2
        self.transform = medical_v1_no_norm_no_tensor
        self.p_rand_order_v1 = 0.2
        self.skip_each_n_slice_mod = 0
        self.epochs = 20
        self.metric = None
        self.stride = 1
        self.skip_each_n_slice = 2
        self.n_slice_per_c = 15
        self.model = MultiInstanceCNNModelRetrain(base_model=timm.create_model(self.model_name, pretrained=True, num_classes=1, in_chans=self.in_chans), n_instance=self.n_slice_per_c, num_classes=len(self.label_features))
        self.use_wandb = False



class class0_gru_chaug_256_2segmodels_v3(rsna_base):
    def __init__(self):
        super().__init__()
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_30_25.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 0]
        self.image_size = 256
        self.in_chans = 3
        self.epochs = 15
        self.model_name = 'senet_mod'
        self.model = Rsna2ndDebugModel(n_instance=15, num_classes=len(self.label_features))
        self.batch_size = 16
        self.grad_accumulations = 1
        self.lr = 0.0001
        self.transform = nodoca_aug_wo_tensor
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')

class class1_gru_128_2segmodels_v3(rsna_base):
    def __init__(self):
        super().__init__()
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_30_25.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 1]
        self.image_size = 128
        self.in_chans = 3
        self.epochs = 15
        self.model_name = 'senet_mod'
        self.model = Rsna2ndDebugModel(n_instance=15, num_classes=len(self.label_features))
        self.batch_size = 16
        self.grad_accumulations = 1
        self.lr = 0.0001
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')

class class1_lstm_112_2segmodels_20epochs_auc_v3(rsna_base):
    def __init__(self):
        super().__init__()
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_30_25.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 1]
        self.image_size = 112
        self.in_chans = 3
        self.epochs = 20
        self.model_name = 'senet_mod'
        self.model = RsnaLstm(n_instance=self.n_slice_per_c, num_classes=len(self.label_features))
        self.batch_size = 16
        self.grad_accumulations = 1
        self.lr = 0.0001
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')
        self.metric = MultiAUC(label_features=self.label_features).torch

class class1_lstm_112_2segmodels_20epochs_auc_v3_pretrained2(rsna_base):
    def __init__(self, fold=0):
        super().__init__()
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_30_25.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 1]
        self.image_size = 112
        self.in_chans = 3
        self.epochs = 20
        self.model_name = 'senet_mod'
        base_model = senet_mod(se_resnext50_32x4d, pretrained=True, num_classes=2)
        if torch.cuda.is_available():
            print('fold:', fold)
            state_dict_path = f'results/rsna_pretrain_seresnext/fold_{fold}.ckpt'
            state_dict = torch.load(state_dict_path)['state_dict']
            torch_state_dict = {}
            for k, v in state_dict.items():
                torch_state_dict[k[6:]] = v
            base_model.load_state_dict(torch_state_dict)
            print(f'base_model load from {state_dict_path}')
        self.model = RsnaLstm(base_model=base_model, n_instance=15, num_classes=len(self.label_features))
        self.batch_size = 16
        self.grad_accumulations = 1
        self.lr = 0.0001
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')
        self.metric = MultiAUC(label_features=self.label_features).torch

class class4_pretrain_288_n25_2segmodels_v3(rsna_base):
    def __init__(self, fold=0):
        super().__init__()
        self.epochs = 20
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_30_25.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 4]
        self.image_size = 288
        self.n_slice_per_c = 25
        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=2, in_chans=self.in_chans)
        if torch.cuda.is_available():
            print('fold:', fold)
            state_dict_path = f'results/rsna_pretrain_convnext/fold_{fold}.ckpt'
            state_dict = torch.load(state_dict_path)['state_dict']
            torch_state_dict = {}
            for k, v in state_dict.items():
                torch_state_dict[k[6:]] = v
            base_model.load_state_dict(torch_state_dict)
            print(f'base_model load from {state_dict_path}')
        self.model = MultiInstanceCNNModelRetrain(base_model=base_model, n_instance=self.n_slice_per_c, num_classes=len(self.label_features))
        self.batch_size //= 2
        self.grad_accumulations *= 2
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')

class class4_pretrain_288_n25_2segmodels_25epochs_v3(rsna_base):
    def __init__(self, fold=0):
        super().__init__()
        self.epochs = 25
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_30_25.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 4]
        self.image_size = 288
        self.n_slice_per_c = 25
        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=2, in_chans=self.in_chans)
        if torch.cuda.is_available():
            print('fold:', fold)
            state_dict_path = f'results/rsna_pretrain_convnext/fold_{fold}.ckpt'
            state_dict = torch.load(state_dict_path)['state_dict']
            torch_state_dict = {}
            for k, v in state_dict.items():
                torch_state_dict[k[6:]] = v
            base_model.load_state_dict(torch_state_dict)
            print(f'base_model load from {state_dict_path}')
        self.model = MultiInstanceCNNModelRetrain(base_model=base_model, n_instance=self.n_slice_per_c, num_classes=len(self.label_features))
        self.batch_size //= 2
        self.grad_accumulations *= 2
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')

class class0_maxvit_224_2segmodels_v4(rsna_base):
    def __init__(self):
        super().__init__()
        self.epochs = 20
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_8_8.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 0]
        self.image_size = 224
        self.model_name = 'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k'
        self.model = MultiInstanceCNNModelRetrain(base_model=timm.create_model(self.model_name, pretrained=True, num_classes=1, in_chans=self.in_chans), n_instance=self.n_slice_per_c, num_classes=len(self.label_features))
        self.batch_size = 4
        self.grad_accumulations = 4
        self.lr = 1e-4
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')

class class23_maxvit_224_2segmodels_v4(rsna_base):
    def __init__(self):
        super().__init__()
        self.epochs = 20
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_8_8.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 34]
        self.image_size = 224
        self.model_name = 'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k'
        self.model = MultiInstanceCNNModelRetrain(base_model=timm.create_model(self.model_name, pretrained=True, num_classes=1, in_chans=self.in_chans), n_instance=self.n_slice_per_c, num_classes=len(self.label_features))
        self.batch_size = 4
        self.grad_accumulations = 4
        self.lr = 1e-4
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')

class class4_pretrain_288_n25_2segmodels_v4(rsna_base):
    def __init__(self, fold=0):
        super().__init__()
        self.epochs = 20
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_8_8.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 4]
        self.image_size = 288
        self.n_slice_per_c = 25
        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=2, in_chans=self.in_chans)
        if torch.cuda.is_available():
            print('fold:', fold)
            state_dict_path = f'results/rsna_pretrain_convnext/fold_{fold}.ckpt'
            state_dict = torch.load(state_dict_path)['state_dict']
            torch_state_dict = {}
            for k, v in state_dict.items():
                torch_state_dict[k[6:]] = v
            base_model.load_state_dict(torch_state_dict)
            print(f'base_model load from {state_dict_path}')
        self.model = MultiInstanceCNNModelRetrain(base_model=base_model, n_instance=self.n_slice_per_c, num_classes=len(self.label_features))
        self.batch_size //= 2
        self.grad_accumulations *= 2
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')

class class4_pretrain_288_n25_2segmodels_25epochs_v4(rsna_base):
    def __init__(self, fold=0):
        super().__init__()
        self.epochs = 25
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_8_8.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 4]
        self.image_size = 288
        self.n_slice_per_c = 25
        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=2, in_chans=self.in_chans)
        if torch.cuda.is_available():
            print('fold:', fold)
            state_dict_path = f'results/rsna_pretrain_convnext/fold_{fold}.ckpt'
            state_dict = torch.load(state_dict_path)['state_dict']
            torch_state_dict = {}
            for k, v in state_dict.items():
                torch_state_dict[k[6:]] = v
            base_model.load_state_dict(torch_state_dict)
            print(f'base_model load from {state_dict_path}')
        self.model = MultiInstanceCNNModelRetrain(base_model=base_model, n_instance=self.n_slice_per_c, num_classes=len(self.label_features))
        self.batch_size //= 2
        self.grad_accumulations *= 2
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')


class class4_lstm_288_n15_2segmodels_v4(rsna_base):
    def __init__(self):
        super().__init__()
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_8_8.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 4]
        self.image_size = 288
        self.in_chans = 3
        self.epochs = 15
        self.model_name = 'senet_mod'
        self.model = RsnaLstm(n_instance=15, num_classes=len(self.label_features))
        self.batch_size = 16
        self.grad_accumulations = 1
        self.lr = 0.0001
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')

class class0_masked_192_2segmodels_pad0(rsna_base):
    def __init__(self):
        super().__init__()
        self.epochs = 20
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_0_0_masked.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 0]
        self.image_size = 192
        self.label_features = ['liver_healthy', 'liver_low', 'liver_high']
        self.in_chans = 3
        self.model = MultiInstanceCNNModelRetrain(base_model=timm.create_model(self.model_name, pretrained=True, num_classes=1, in_chans=self.in_chans), n_instance=15, num_classes=len(self.label_features))
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')

class class0_masked_192_2segmodels_30epochs_pad0(rsna_base):
    def __init__(self):
        super().__init__()
        self.epochs = 30
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_0_0_masked.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 0]
        self.image_size = 192
        self.label_features = ['liver_healthy', 'liver_low', 'liver_high']
        self.in_chans = 3
        self.model = MultiInstanceCNNModelRetrain(base_model=timm.create_model(self.model_name, pretrained=True, num_classes=1, in_chans=self.in_chans), n_instance=15, num_classes=len(self.label_features))
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')

class class23_maxvit_224_2segmodels_v4_pretrained(rsna_base):
    def __init__(self, fold=0):
        super().__init__()
        self.epochs = 20
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_8_8.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 34]
        self.image_size = 224
        self.model_name = 'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k'
        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=2, in_chans=self.in_chans)
        if torch.cuda.is_available():
            print('fold:', fold)
            state_dict_path = f'results/rsna_pretrain_maxvit/fold_{fold}.ckpt'
            state_dict = torch.load(state_dict_path)['state_dict']
            torch_state_dict = {}
            for k, v in state_dict.items():
                torch_state_dict[k[6:]] = v
            base_model.load_state_dict(torch_state_dict)
            print(f'base_model load from {state_dict_path}')
        self.model = MultiInstanceCNNModelRetrain(base_model=base_model, n_instance=self.n_slice_per_c, num_classes=len(self.label_features))
        self.batch_size = 4
        self.grad_accumulations = 4
        self.lr = 1e-4
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')

class class0_gru_chaug_224_2segmodels_v3_caformer_v2(rsna_base):
    def __init__(self):
        super().__init__()
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_30_25.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 0]
        self.image_size = 224
        self.in_chans = 3
        self.epochs = 15
        self.model_name = 'caformer_s36.sail_in22k_ft_in1k'
        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=1, in_chans=self.in_chans)
        self.model = RSNA2ndModel2(base_model=base_model, n_instance=15, num_classes=len(self.label_features), pool='avg')
        self.batch_size = 4
        self.grad_accumulations = 4
        self.lr = 0.0001
        self.transform = nodoca_aug_wo_tensor
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')

class class0_masked_192_2segmodels_pad0_caformer(rsna_base):
    def __init__(self):
        super().__init__()
        self.epochs = 20
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_0_0_masked.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 0]
        self.image_size = 192
        self.label_features = ['liver_healthy', 'liver_low', 'liver_high']
        self.in_chans = 3
        self.model_name = 'caformer_s36.sail_in22k_ft_in1k'
        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=1, in_chans=self.in_chans)
        self.model = RSNA2ndModel2(base_model=base_model, n_instance=15, num_classes=len(self.label_features), pool='avg')
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        if 'fold' in list(self.train_df):
            del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')
        self.batch_size = 4
        self.grad_accumulations = 4

class class23_2segmodels_v3_xcit_small(rsna_2nd_class0_v1):
    def __init__(self):
        super().__init__()
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_30_25.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class == 34]
        self.image_size = 192
        self.in_chans = 3
        self.epochs = 15
        self.model_name = 'xcit_small_24_p8_224.fb_dist_in1k'
        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=1, in_chans=self.in_chans)
        self.model = RsnaLstmXcit(base_model=base_model, n_instance=15, num_classes=len(self.label_features))
        self.batch_size = 4
        self.grad_accumulations = 4
        self.lr = 0.0001
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')

class class_all_pad30_maxvit_pretrained(rsna_2nd_class0_v1):
    def __init__(self, fold=0):
        super().__init__()
        self.epochs = 8
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_30_25.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class.isin([0,1,34,4])]
        self.class_image_size_map = {'0': 224, '1': 224, '23': 224, '4': 224}
        self.class_n_instance_map = {'0': 15, '1': 15, '23': 15, '4': 25}
        self.model_name = 'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k'
        self.batch_size = 2
        self.grad_accumulations = 8
        self.transform = nodoca_aug_wo_tensor_wo_resize
        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=2, in_chans=self.in_chans)
        if torch.cuda.is_available():
            print('fold:', fold)
            state_dict_path = f'../results/rsna_for_extravasation_oof_maxvit_1e_5/fold_{fold}.ckpt'
            state_dict = torch.load(state_dict_path)['state_dict']
            torch_state_dict = {}
            for k, v in state_dict.items():
                torch_state_dict[k[6:]] = v
            base_model.load_state_dict(torch_state_dict)
            print(f'base_model load from {state_dict_path}')
        self.model = MultiInstanceCNNModelRetrain(base_model=base_model, n_instance=self.n_slice_per_c, num_classes=len(self.label_features))
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')
        self.use_custom_sampler = True
        self.resize_by_class = True
        self.transform = medical_v1_no_norm_no_tensor_no_resize
        self.gpu = 'small'
        self.use_wandb = True
        self.lr = 1e-4

class class_all_pad30_maxvit_crop2_pretrained(rsna_2nd_class0_v1):
    def __init__(self, fold=0):
        super().__init__()
        self.epochs = 8
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_8_8.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class.isin([0,1,34,4])]
        self.class_image_size_map = {'0': 224, '1': 224, '23': 224, '4': 224}
        self.class_n_instance_map = {'0': 15, '1': 15, '23': 15, '4': 25}
        self.batch_size = 2
        self.grad_accumulations = 8
        self.transform = nodoca_aug_wo_tensor_wo_resize
        self.model_name = 'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k'
        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=2, in_chans=self.in_chans)
        if torch.cuda.is_available():
            print('fold:', fold)
            state_dict_path = f'../results/rsna_for_extravasation_oof_maxvit_1e_5/fold_{fold}.ckpt'
            state_dict = torch.load(state_dict_path)['state_dict']
            torch_state_dict = {}
            for k, v in state_dict.items():
                torch_state_dict[k[6:]] = v
            base_model.load_state_dict(torch_state_dict)
            print(f'base_model load from {state_dict_path}')
        self.model = MultiInstanceCNNModelRetrain(base_model=base_model, n_instance=self.n_slice_per_c, num_classes=len(self.label_features))
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')
        self.use_custom_sampler = True
        self.resize_by_class = True
        self.gpu = 'small'
        self.use_wandb = True
        self.lr = 1e-4
class class_all_pad30_convnext_pretrained(rsna_2nd_class0_v1):
    def __init__(self, fold=0):
        super().__init__()
        self.epochs = 8
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_30_25.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class.isin([0,1,34,4])]
        self.class_image_size_map = {'0': 248, '1': 112, '23': 184, '4': 280}
        self.class_n_instance_map = {'0': 15, '1': 15, '23': 10, '4': 25}
        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=2, in_chans=self.in_chans)
        if torch.cuda.is_available():
            print('fold:', fold)
            state_dict_path = f'../results/rsna_for_extravasation_oof_v2/fold_{fold}.ckpt'
            state_dict = torch.load(state_dict_path)['state_dict']
            torch_state_dict = {}
            for k, v in state_dict.items():
                torch_state_dict[k[6:]] = v
            base_model.load_state_dict(torch_state_dict)
            print(f'base_model load from {state_dict_path}')
        self.model = RSNA2ndModel2(base_model=base_model, n_instance=15, num_classes=len(self.label_features), pool='avg')
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')
        self.use_custom_sampler = True
        self.resize_by_class = True
        self.transform = medical_v1_no_norm_no_tensor_no_resize
        self.gpu = 'small'
        self.use_wandb = True
class class_all_pad30_maxvit_pretrained(rsna_2nd_class0_v1):
    def __init__(self, fold=0):
        super().__init__()
        self.epochs = 8
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_30_25.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class.isin([0,1,34,4])]
        self.class_image_size_map = {'0': 224, '1': 224, '23': 224, '4': 224}
        self.class_n_instance_map = {'0': 15, '1': 15, '23': 15, '4': 25}
        self.model_name = 'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k'
        self.batch_size = 2
        self.grad_accumulations = 8
        self.transform = nodoca_aug_wo_tensor_wo_resize
        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=2, in_chans=self.in_chans)
        if torch.cuda.is_available():
            print('fold:', fold)
            state_dict_path = f'../results/rsna_for_extravasation_oof_maxvit_1e_5/fold_{fold}.ckpt'
            state_dict = torch.load(state_dict_path)['state_dict']
            torch_state_dict = {}
            for k, v in state_dict.items():
                torch_state_dict[k[6:]] = v
            base_model.load_state_dict(torch_state_dict)
            print(f'base_model load from {state_dict_path}')
        self.model = MultiInstanceCNNModelRetrain(base_model=base_model, n_instance=self.n_slice_per_c, num_classes=len(self.label_features))
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')
        self.use_custom_sampler = True
        self.resize_by_class = True
        self.transform = medical_v1_no_norm_no_tensor_no_resize
        self.gpu = 'small'
        self.use_wandb = True
        self.lr = 1e-4
class class_all_pad30_caformer(rsna_2nd_class0_v1):
    def __init__(self):
        super().__init__()
        self.epochs = 8
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_30_25.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df.image_id.nunique()
        self.train_df = self.train_df[self.train_df.mask_class.isin([0,1,34,4])]
        self.class_image_size_map = {'0': 256, '1': 128, '23': 192, '4': 256}
        self.class_n_instance_map = {'0': 15, '1': 15, '23': 15, '4': 25}
        self.model_name = 'caformer_s36.sail_in22k_ft_in1k'
        self.model = RSNA2ndModel(base_model=timm.create_model(self.model_name, pretrained=True, num_classes=1, in_chans=self.in_chans), n_instance=15, num_classes=len(self.label_features), pool='avg')
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')
        self.use_custom_sampler = True
        self.resize_by_class = True
        self.transform = medical_v1_no_norm_no_tensor_no_resize
        self.gpu = 'small'
        self.use_wandb = True
        self.batch_size = 4
        self.grad_accumulations = 4
class class_all_pad30_seresnext_crop2_pretrained(rsna_2nd_class0_v1):
    def __init__(self, fold=0):
        super().__init__()
        self.epochs = 8
        self.train_df_path = '../input/train2.5d_2models_th0.2_pad_8_8.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.mask_class.isin([0,1,34,4])]
        self.class_image_size_map = {'0': 256, '1': 128, '23': 192, '4': 288}
        self.class_n_instance_map = {'0': 15, '1': 15, '23': 15, '4': 30}
        self.model_name = 'senet_mod'
        base_model = senet_mod(se_resnext50_32x4d, pretrained=True, num_classes=2)
        if torch.cuda.is_available():
            print('fold:', fold)
            state_dict_path = f'../results/rsna_for_extravasation_oof_seresnext_stride2_fixlr/fold_{fold}.ckpt'
            state_dict = torch.load(state_dict_path)['state_dict']
            torch_state_dict = {}
            for k, v in state_dict.items():
                torch_state_dict[k[6:]] = v
            base_model.load_state_dict(torch_state_dict)
            print(f'base_model load from {state_dict_path}')
        self.model = RSNA2ndModel(base_model = base_model, n_instance=15, num_classes=len(self.label_features), pool='avg')
        fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
        del self.train_df['fold']
        self.train_df = self.train_df.merge(fold, on='patient_id')
        self.use_custom_sampler = True
        self.resize_by_class = True
        self.transform = medical_v1_no_norm_no_tensor_no_resize
        self.gpu = 'small'
        self.use_wandb = True
        self.lr = 1e-4

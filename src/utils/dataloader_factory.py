from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
from transformers import RobertaConfig, RobertaModel, XLNetTokenizer, AutoTokenizer
import torch
import sys
import os
sys.path.append(os.path.abspath(".."))
from src.lightning.data_modules.classification import get_dataset_class
from src.lightning.data_modules.segmentation import SegmentationDataset, SegmentationDataset3D
from src.lightning.data_modules.mlp_with_nlp import MlpWithNlpDataset
from src.lightning.data_modules.mlp import MLPDataset
from src.lightning.data_modules.nlp import NLPDataset
from src.lightning.data_modules.mil import MilDataset

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# class DatasetTest(Dataset):
#     def __init__(self, df, transforms, cfg):
#         self.transforms = transforms
#         self.paths = df.path.values

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         path = self.paths[idx]
#         image = cv2.imread(path)[:,:,::-1]
#         if self.cfg.box_crop:
#             box = self.boxes[idx]
#             image = image[box[1]:box[3], box[0]:box[2], :]

#         if self.cfg.pad_square:
#             image = pad_to_square(image, self.cfg.image_size[1]/self.cfg.image_size[0])

#         if self.transforms:
#             image = self.transforms(image=image)['image']
#         if image.shape[2] < 10:
#             image = image.transpose(2, 0, 1)

#         return image

# class EffdetDatasetTest(Dataset):
#     def __init__(self, df, transforms, cfg):
#         self.transforms = transforms
#         self.paths = df.path.unique()
#         self.df = df
#         self.cfg = cfg

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx: int):
#         path = self.paths[idx]

#         image = self.load_image(idx)

#         if self.transforms:
#             image = self.transforms(image=image)['image']
#         if image.shape[2] < 10:
#             image = image.transpose(2, 0, 1)

#         return image

#     def load_image(self, idx):
#         path = self.paths[idx]
#         image = cv2.imread(path, cv2.IMREAD_COLOR).copy()
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
#         image /= 255.0
#         return image

def prepare_classification_loader(cfg, split='val'):
    if split == 'train':
        df = cfg.train_df[cfg.train_df.fold != cfg.fold]
    elif split == 'val':
        df = cfg.train_df[cfg.train_df.fold == cfg.fold]
    elif split == 'test':
        df = cfg.test_df
    else:
        raise
    claz = get_dataset_class(cfg)
    ds = claz(
        df=df,
        transforms=cfg.transform['val'],
        cfg=cfg,
        phase='test'
    )


    return df, DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                      num_workers=cfg.n_cpu, worker_init_fn=worker_init_fn)

def prepare_mlp_loader(cfg, split='val'):
    if split == 'train':
        df = cfg.train_df[cfg.train_df.fold != cfg.fold]
    elif split == 'val':
        df = cfg.train_df[cfg.train_df.fold == cfg.fold]
    elif split == 'test':
        df = cfg.test_df
    else:
        raise

    ds = MLPDataset(
        df=df,
        cfg=cfg,
        phase='test'
    )

    return df, DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                      num_workers=cfg.n_cpu, worker_init_fn=worker_init_fn)

def prepare_mlp_with_nlp_loader(cfg, split='val'):
    if split == 'train':
        df = cfg.train_df[cfg.train_df.fold != cfg.fold]
    elif split == 'val':
        df = cfg.train_df[cfg.train_df.fold == cfg.fold]
    elif split == 'test':
        df = cfg.test_df
    else:
        raise

    ds = MlpWithNlpDataset(
        df=df,
        cfg=cfg,
        phase='test'
    )
    return df, DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                      num_workers=cfg.n_cpu, worker_init_fn=worker_init_fn)

def prepare_seg_loader(cfg, split='val'):
    if split == 'train':
        df = cfg.train_df[cfg.train_df.fold != cfg.fold]
    elif split == 'val':
        df = cfg.train_df[cfg.train_df.fold == cfg.fold]
    elif split == 'test':
        df = cfg.test_df
    else:
        raise

    claz = SegmentationDataset3D if cfg.seg_3d else SegmentationDataset
    ds = claz(
        df=df,
        transforms=cfg.transform['val'],
        cfg=cfg,
        phase='test'
    )
    return df, DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                      num_workers=cfg.n_cpu, worker_init_fn=worker_init_fn)

def prepare_effdet_loader(cfg, predict_valid=False):
    if predict_valid:
        df = cfg.train_df[cfg.train_df.fold == cfg.fold]
        image = cv2.imread(path)[:,:,::-1]

    ds = EffdetDatasetTest(
        df=df,
        transforms=cfg.transform['test'],
        cfg=cfg
    )
    return df, DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                      num_workers=cfg.n_cpu, worker_init_fn=worker_init_fn)

def prepare_nlp_loader(cfg, predict_valid=False):
    df = cfg.train_df[cfg.train_df.fold == cfg.fold] if predict_valid else cfg.test_df
    ds = NLPDataset(
        df=df,
        cfg=cfg,
        phase='test',
    )
    return df, DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                      num_workers=cfg.n_cpu, worker_init_fn=worker_init_fn)

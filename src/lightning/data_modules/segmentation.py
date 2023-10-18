import numpy as np
import pandas as pd
import json
from pathlib import Path
import pickle
from glob import glob
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
from pdb import set_trace as st

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .util import *
import os

from multiprocessing import Pool, cpu_count
# resize_transform = A.Compose([A.Resize(height=self.cfg.image_size[0], width=self.cfg.image_size[1], p=1.0)])
from monai.transforms import Resize
import monai

# ForkedPdb().set_trace()
import pdb
import sys
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def rsna_load_img(path, cfg):
    strides = cfg.strides
    image_size = cfg.image_size
    im = cv2.imread(path)[:,:,0]
    im = cv2.resize(im, image_size)
    img = [im]
    for stride in strides:
        prev_path, next_path = rsna_get_prev_next_path_v2(path, stride)
        # print('\n',prev_path, path, next_path,'\n')
        prev = cv2.imread(prev_path)[:,:,0]
        prev = cv2.resize(prev, image_size)
        img.append(prev)
        next_ = cv2.imread(next_path)[:,:,0]
        next_ = cv2.resize(next_, image_size)
        img.append(next_)
    img = np.array(img).transpose(1,2,0)
    # img = img.astype('float32') # original is uint16
    return img#.astype('uint8')

def rsna_get_prev_next_path_v2(path, stride):
    id = path.split('_')[-1]
    path_base = '_'.join(path.split('_')[:-1])
    origin_slice = int(id.split('.')[0])
    prev = origin_slice - stride
    prev_path = f'{path_base}_{str(prev).zfill(4)}.png'
    for i in range(stride):
        if os.path.exists(prev_path):
            break
        else:
            prev+=1
            prev_path = f'{path_base}_{str(prev).zfill(4)}.png'

    next_ = origin_slice + stride
    next_path = path.replace(path.split('/')[-1], f'{str(next_).zfill(4)}.png')
    next_path = f'{path_base}_{str(next_).zfill(4)}.png'
    for i in range(stride):
        if os.path.exists(next_path):
            break
        else:
            next_-=1
            next_path = f'{path_base}_{str(next_).zfill(4)}.png'
    return prev_path, next_path


def rsna_get_prev_next_path(path, stride):
    id = path.split('/')[-1]
    origin_slice = int(id.split('.')[0])
    prev = origin_slice - stride
    prev_path = path.replace(path.split('/')[-1], f'{str(prev).zfill(4)}.npy')
    for i in range(stride):
        if os.path.exists(prev_path):
            break
        else:
            prev+=1
            prev_path = path.replace(path.split('/')[-1], f'{str(prev).zfill(4)}.npy')

    next_ = origin_slice + stride
    next_path = path.replace(path.split('/')[-1], f'{str(next_).zfill(4)}.npy')
    for i in range(stride):
        if os.path.exists(next_path):
            break
        else:
            next_-=1
            next_path = path.replace(path.split('/')[-1], f'{str(next_).zfill(4)}.npy')
    return prev_path, next_path


class SegmentationDataset(Dataset):
    def __init__(self, df, transforms, cfg, phase):
        self.transforms = transforms
        self.paths = df.path.values
        self.cfg = cfg
        self.phase = phase
        if self.phase == 'test':
            self.ids = df.image_id.values
        if phase != 'test':
            self.mask_paths = df.mask_path.values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if not os.path.exists(path):
            print('not exists:', path)
            raise

        if getattr(self.cfg, 'strides', False):
            image = rsna_load_img(path, self.cfg)
        else:
            image = cv2.imread(path)[:,:,::-1]
        # image = np.load(path)

        h, w = image.shape[:2]

        if self.phase!='test':
            mask_path = self.mask_paths[idx]
            mask = np.load(mask_path)
            if len(mask.shape) == 2:
                mask = np.array([mask]).transpose((1,2,0))
            if 255.0 in mask:
                mask = (mask/255)
            if getattr(self.cfg, 'mask_transpose', False):
                mask = mask.transpose((1,2,0))
            mask = mask.astype(np.uint8)

        if self.transforms:
            if self.phase == 'test':
                image = self.transforms(image=image)['image']
            else:
                aug = self.transforms(image=image, mask=mask)
                image = aug['image']
                mask = aug['mask']
        if self.phase == 'test':
            id_ = self.ids[idx]
            return image, id_#, h, w
        else:
            mask = np.transpose(mask, (2, 0, 1))
            return image, mask

class SegmentationDataset3D(Dataset):
    def __init__(self, df, transforms, cfg, phase):
        self.paths = df.path.values
        self.cfg = cfg
        self.phase = phase
        if self.phase == 'test':
            self.ids = df.image_id.values
        if phase != 'test':
            self.mask_paths = df.mask_path.values

        if self.phase == 'train':
            self.transforms = monai.transforms.Compose([
                monai.transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
                monai.transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
                monai.transforms.RandAffined(keys=["image", "mask"], translate_range=[int(x*y) for x, y in zip(self.cfg.image_size, [0.3, 0.3, 0.3])], padding_mode='zeros', prob=0.7),
                monai.transforms.RandGridDistortiond(keys=("image", "mask"), prob=0.5, distort_limit=(-0.01, 0.01), mode="nearest"),
            ])

        self.resize = Resize(cfg.image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if not os.path.exists(path):
            print('not exists:', path)
            raise

        image = np.load(path) # (z, y, x)
        if getattr(self.cfg, 'skip_each_n_slice', False):
            ids = [i for i in range(len(image)) if i % self.cfg.skip_each_n_slice == 0]
            image = image[ids]

        if self.phase != 'test':
            mask_path = self.mask_paths[idx]
            mask = np.load(mask_path) # (num_classes, z, y, x)
            if getattr(self.cfg, 'skip_each_n_slice', False):
                mask = mask[:, ids, :, :]
                assert image.shape == mask[0].shape

            if np.max(mask) == 1:
                mask *= 255
            if mask.ndim < 4:
                mask = np.expand_dims(mask, 0)
            # print('mask.shape:',mask.shape)
            mask = self.resize(mask)#.numpy()

        # print('image.shape, mask.shape:', image.shape, mask.shape)
        image = image[np.newaxis]
        image = self.resize(image)#.numpy()

        if image.ndim < 4:
            image = np.expand_dims(image, 0)
        image = image.astype(np.float32).repeat(3, 0)  # to 4ch

        if self.phase == 'train':
            res = self.transforms({'image':image, 'mask':mask})
            image = res['image'] / 255.
            mask = res['mask']
        else:
            image = image / 255.
        image = torch.tensor(image).float()
        if self.phase == 'test':
            id_ = self.ids[idx]
            return image, id_#, h, w
        else:
            mask = (mask > 127).astype(np.float32)
            mask = torch.tensor(mask).float()
            # print(path, 'image.size(), mask.size():', image.size(), mask.size())
            return image, mask


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    # 必ず呼び出される関数
    def setup(self, stage):
        pass

    # Trainer.fit() 時に呼び出される
    def train_dataloader(self):
        if self.cfg.train_by_all_data:
            tr = self.cfg.train_df
        else:
            tr = self.cfg.train_df[self.cfg.train_df.fold != self.cfg.fold]
        # tr=tr.sample(100)
        claz = SegmentationDataset3D if self.cfg.seg_3d else SegmentationDataset
        train_ds = claz(
            df=tr,
            transforms=self.cfg.transform['train'],
            cfg=self.cfg,
            phase='train'
        )
        return DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True,
                          num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn)

    def get_val(self):
        val = self.cfg.train_df[self.cfg.train_df.fold == self.cfg.fold]
        if ('type' in list(val)) and ('origin' in val['type'].unique()):
            val = val[val.type=='origin']
        return val

    # Trainer.fit() 時に呼び出される
    def val_dataloader(self):
        val = self.get_val()

        claz = SegmentationDataset3D if self.cfg.seg_3d else SegmentationDataset
        valid_ds = claz(
            df=val,
            transforms=self.cfg.transform['val'],
            cfg=self.cfg,
            phase='valid'
        )
        return DataLoader(valid_ds, batch_size=self.cfg.batch_size, shuffle=False, drop_last=False,
                          num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn)

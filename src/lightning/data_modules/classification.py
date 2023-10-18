import numpy as np
import pandas as pd
import json
from pathlib import Path
import pickle
from glob import glob
import cv2
from PIL import Image
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pdb import set_trace as st

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pytorch_lightning as pl
from .util import *
from .mil import MilDataset
import os

from multiprocessing import Pool, cpu_count
from pfio.cache import MultiprocessFileCache
from monai.transforms import Resize
from albumentations import ReplayCompose

def sigmoid(x):
    return 1/(1 + np.exp(-x))

import pdb
import sys
# ForkedPdb().set_trace()
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

def pad_to_square(a, wh_ratio=4):
    if len(a.shape) == 2:
        a = np.array([a,a,a]).transpose(1,2,0)
        grayscale = True
    else:
        grayscale = False

    """ Pad an array `a` evenly until it is a square """
    if a.shape[1]>a.shape[0]*wh_ratio: # pad height
        n_to_add = a.shape[1]/wh_ratio-a.shape[0]

        pad = int(n_to_add//2)
        # bottom_pad = int(n_to_add-top_pad)
        a = np.pad(a, [(pad, pad), (0, 0), (0, 0)], mode='constant')

    elif a.shape[0]*wh_ratio>a.shape[1]: # pad width
        n_to_add = a.shape[0]*wh_ratio-a.shape[1]
        pad = int(n_to_add//2)
        # right_pad = int(n_to_add-left_pad)
        a = np.pad(a, [(0, 0), (pad, pad), (0, 0)], mode='constant')
    if grayscale:
        a = a[:,:,0]
    return a

def slice_image(img):
    _, width,ch = img.shape
    new_width = int(width / 2)
    left_img = img[:, 0:new_width]
    right_img = img[:, new_width:]
    return left_img, right_img

def mask_pixels(img):
    img[img > 1] = 1
    return img

def count_pixels(img):
    return np.sum(img)

def normalize_horiz_orientation(img, reverse=False):
    left_img, right_img = slice_image(mask_pixels(img.copy()))
    if reverse:
        if count_pixels(right_img) <= count_pixels(left_img):
            return cv2.flip(img, 1)
    else:
        if count_pixels(right_img) > count_pixels(left_img):
            return cv2.flip(img, 1)
    return img

# resize_transform = A.Compose([A.Resize(height=self.cfg.image_size[0], width=self.cfg.image_size[1], p=1.0)])
def load_image(args):
    path, imsize = args
    image = cv2.imread(path)[:,:,::-1]
    # 画像を拡大する場合は、 INTER_LINEARまたはINTER_CUBIC補間を使用することをお勧めします。画像を縮小する場合は、 INTER_AREA補間を使用することをお勧めします。
    # キュービック補間は計算が複雑であるため、線形補間よりも低速です。ただし、結果の画像の品質は高くなります。
    return path, cv2.resize(image, (imsize[1], imsize[0]), interpolation=cv2.INTER_AREA)

def rsna_load_img(path, cfg):
    strides = cfg.strides
    image_size = cfg.image_size
    im = cv2.imread(path)[:,:,0]
    # im = cv2.resize(im, image_size)
    img = [im]
    for stride in strides:
        prev_path, next_path = rsna_get_prev_next_path_v2(path, stride)
        # print('\n',prev_path, path, next_path,'\n')
        try:
            prev = cv2.imread(prev_path)[:,:,0]
        except:
            prev = np.zeros(im.shape)
        # prev = cv2.resize(prev, image_size)
        img.append(prev)
        try:
            next_ = cv2.imread(next_path)[:,:,0]
        except:
            next_ = np.zeros(im.shape)

        # next_ = cv2.resize(next_, image_size)
        img.append(next_)
    img = np.array(img).transpose(1,2,0)
    # img = img.astype('float32') # original is uint16
    return img.astype('uint8')

def rsna_get_prev_next_path_v2(path, stride):
    id = path.split('_')[-1]
    path_base = '_'.join(path.split('_')[:-1])
    origin_slice = int(id.split('.')[0])
    prev = origin_slice - stride
    prev_path = f'{path_base}_{str(prev).zfill(4)}.png'
    # for i in range(stride):
    #     if os.path.exists(prev_path):
    #         break
    #     else:
    #         prev+=1
    #         prev_path = f'{path_base}_{str(prev).zfill(4)}.png'

    next_ = origin_slice + stride
    next_path = path.replace(path.split('/')[-1], f'{str(next_).zfill(4)}.png')
    next_path = f'{path_base}_{str(next_).zfill(4)}.png'
    # for i in range(stride):
    #     if os.path.exists(next_path):
    #         break
    #     else:
    #         next_-=1
    #         next_path = f'{path_base}_{str(next_).zfill(4)}.png'
    return prev_path, next_path


class ClassificationDataset(Dataset):
    def __init__(self, df, transforms, cfg, phase, current_epoch=None):
        self.transforms = transforms
        self.paths = df.path.values
        self.cfg = cfg
        self.phase = phase
        if phase != 'test':
            self.labels = df[cfg.label_features].values

    def __len__(self):
        return len(self.paths)

    def _read_image(self, path):
        if not os.path.exists(path):
            print('not exists:', path)
            raise

        if getattr(self.cfg, 'strides', False):
            image = rsna_load_img(path, self.cfg)
        else:
            if '.npy' in path:
                image = np.load(path)
                image = np.array([image, image, image]).transpose((1,2,0))
            else:
                image = cv2.imread(path)[:,:,::-1]
        return image, 0

    def __getitem__(self, idx):
        path = self.paths[idx]

        image, _ = self._read_image(path)

            if self.transforms:
                image = self.transforms(image=image)['image']
            if image.shape[2] < 10:
                image = image.transpose(2, 0, 1)

        if len(self.cfg.meta_cols) != 0:
            meta = torch.FloatTensor(self.metas[idx])
            image = (image, meta)

        if self.phase == 'test':
            return image

        label = self.labels[idx]

        if type(self.cfg.label_features) == list:
            return image, torch.FloatTensor(label) # multi class
        else:
            if str(self.cfg.criterion) == 'CrossEntropyLoss()':
                return image, label
            return image, label.astype(np.float32)

def select_numbers_around_m(N, m, in_chans, stride):
    results = []
    for i in range(in_chans):
        i-=(in_chans-1)//2
        i*=stride
        if i < 0:
            results.append(max(0, m+i))
        else:
            results.append(min(N, m+i))
    return results

def get_middle_numbers(N, num_groups=15):
    bins = np.linspace(0, N, num_groups+1, dtype=int)

    middle_nums = []
    # 各グループで中央の値を取得
    for i in range(len(bins)-1):
        start = bins[i]
        end = bins[i+1]
        mid = (start + end) // 2
        middle_nums.append(mid)

    return middle_nums

def resize_with_padding(img, target_shape=(256, 256, 3)):
    # 黒色で初期化された目標shapeの空の画像を作成
    background = np.zeros(target_shape, dtype=np.uint8)

    # 入力された画像のshapeを取得
    h, w = img.shape[:2]

    # 入力された画像を背景の左上隅に配置
    background[:h, :w] = img

    return background

import random
import time
import numpy as np
from torch.utils.data import Sampler
class InterleavedMaskClassBatchSampler(Sampler):
    def __init__(self, df, cfg):
        self.df = df
        self.batch_size = cfg.batch_size
        self.indices_by_class = {cls: list(df[df['mask_class'] == cls].index) for cls in df['mask_class'].unique()}
        for indices in self.indices_by_class.values():
            np.random.shuffle(indices)

    def __iter__(self):
        all_classes = list(self.indices_by_class.keys())
        while len(all_classes) > 0:
            cls = np.random.choice(all_classes)
            if len(self.indices_by_class[cls]) >= self.batch_size:
                print(self.indices_by_class[cls][:self.batch_size])
                for _ in range(self.batch_size):
                    yield self.indices_by_class[cls].pop()
            else:
                all_classes.remove(cls)

    def __len__(self):
        return sum(len(indices) for indices in self.indices_by_class.values())

class InterleavedMaskClassBatchSampler(Sampler):
    def __init__(self, df, cfg):

        df['tmp_for_batch_sampler'] = list(range(len(df)))
        self.batch_size = cfg.batch_size
        self.df = df
        self.init_indices()
        self.total_len = len(self.batch_indices_list) * self.batch_size
        # st()
        # df[df.tmp_for_batch_sampler.isin(self.batch_indices_list[1])]

    def init_indices(self):
        chunks = []
        for c in [0, 1, 34, 4]:
            cdf = self.df[self.df.mask_class == c]
            cdf = cdf.sample(len(cdf))
            lst = cdf.tmp_for_batch_sampler.values.tolist()
            chunks += [lst[i:i+self.batch_size] for i in range(0, len(lst), self.batch_size) if len(lst[i:i+self.batch_size]) == self.batch_size]

        self.batch_indices_list = random.sample(chunks, len(chunks))


    def __iter__(self):
        for batch_indices in self.batch_indices_list:
            for idx in batch_indices:
                yield idx

    def __len__(self):
        return self.total_len

class InterleavedMaskClassBatchSamplerBK(Sampler):
    def __init__(self, df, cfg):
        self.df = df
        self.batch_size = cfg.batch_size
        self.init_indices()

    def init_indices(self):
        self.indices_by_class = {cls: list(self.df[self.df['mask_class'] == cls].index) for cls in self.df['mask_class'].unique()}
        for indices in self.indices_by_class.values():
            np.random.shuffle(indices)

    def __iter__(self):
        self.init_indices()  # 各エポックの開始時にインデックスを再初期化
        all_classes = list(self.indices_by_class.keys())
        while len(all_classes) > 0:
            cls = np.random.choice(all_classes)
            while len(self.indices_by_class[cls]) >= self.batch_size:
                for _ in range(self.batch_size):
                    yield self.indices_by_class[cls].pop()
            if len(self.indices_by_class[cls]) < self.batch_size and len(self.indices_by_class[cls]) > 0:
                for _ in range(len(self.indices_by_class[cls])):
                    yield self.indices_by_class[cls].pop()
            all_classes.remove(cls)

    def __len__(self):
        return sum(len(indices) for indices in self.indices_by_class.values())

def collate_fn(batch):
    images, targets= list(zip(*batch))
    # images = torch.stack(images)
    # targets = torch.stack(targets)
    return images, targets

class ClassificationDatasetMultiImage(Dataset):
    def __init__(self, df, transforms, cfg, phase, current_epoch=None):
        self.transforms = transforms
        self.paths = df.path.values
        self.cfg = cfg
        self.phase = phase
        if phase != 'test':
            self.labels = df[cfg.label_features].values
        if hasattr(cfg, 'meta_df'):
            self.meta_df = cfg.meta_df
        self.mask_classes = df[['mask_class']].values
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # path = [path for path in self.paths if '10026_42932' in path][0]
        # print(path)
        # images = np.load(path, allow_pickle=True)
        images = np.load(self.paths[idx], allow_pickle=True)
        if getattr(self.cfg, 'resize_by_class', False):
            mask_class = self.paths[idx].split('_')[-1].replace('.npy', '')
            self.cfg.n_slice_per_c = self.cfg.class_n_instance_map[mask_class]

        if getattr(self.cfg, 'skip_each_n_slice', False):
            images = images[[i for i in range(len(images)) if i % self.cfg.skip_each_n_slice == self.cfg.skip_each_n_slice_mod]]
        total_images_len = len(images)

        if getattr(self.cfg, 'include_black_images', False):
            if len(images) >= self.cfg.n_slice_per_c:
                indexes = get_middle_numbers(len(images), num_groups=self.cfg.n_slice_per_c)
            else:
                indexes = list(range(len(images)))
        else:
            indexes = get_middle_numbers(len(images), num_groups=self.cfg.n_slice_per_c)

        ids_list = []
        if getattr(self.cfg, 'input_3ch', False):
            ids_list = indexes
            all_image_num = self.cfg.n_slice_per_c
            # print(ids_list)
        elif getattr(self.cfg, 'equal_sample', False):
            ids_list = [indexes[i:i+self.cfg.in_chans] for i in range(0, len(indexes), self.cfg.in_chans)]
            all_image_num = self.cfg.n_slice_per_c//self.cfg.in_chans
        else:
            for i in indexes:
                ids = select_numbers_around_m(len(images)-1, i, self.cfg.in_chans, self.cfg.stride)
                ids_list.append(ids)
            all_image_num = self.cfg.n_slice_per_c

        transformed_images = []
        replay = None
        for ids in ids_list:
            image = images[ids]
            image = image.transpose((1,2,0))

            if getattr(self.cfg, 'resize_by_class', False):
                mask_class = self.paths[idx].split('_')[-1].replace('.npy', '')
                image_size = self.cfg.class_image_size_map[mask_class]
                image = cv2.resize(image, (image_size, image_size))
            image = self.transforms(image=image)['image']
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.
            transformed_images.append(image)

        images = np.stack(transformed_images, 0)
        images = torch.tensor(images).float()
        if len(images) < all_image_num:
            n_instance_current, channel, width, heigth = images.size()

            images = torch.cat(
                [images, torch.zeros(all_image_num - n_instance_current, channel, width, heigth)]
            )

        if self.phase == 'train' and random.random() < self.cfg.p_rand_order_v1:
            indices = torch.randperm(images.size(0))
            images = images[indices]

        if self.phase == 'test':
            return images

        label = self.labels[idx]

        if type(self.cfg.label_features) == list:
            return images, torch.FloatTensor(label) # multi class
        else:
            if str(self.cfg.criterion) == 'CrossEntropyLoss()':
                return images, label
            # return images, torch.FloatTensor(label) # multi class
            return images, label.astype(np.float32)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_dataset_class(cfg):
    if getattr(cfg, 'multi_image', False):
        claz = ClassificationDatasetMultiImage
    else:
        claz = ClassificationDataset
    return claz

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
        if self.cfg.upsample is not None:
            assert type(self.cfg.upsample) == int
            origin_len = len(tr)
            dfs = [tr]
            for col in self.cfg.label_features:
                for _ in range(self.cfg.upsample):
                    dfs.append(tr[tr[col]==1])
            tr = pd.concat(dfs)
            print(f'upsample, len: {origin_len} -> {len(tr)}')

        print('len(train):', len(tr))
        claz = get_dataset_class(self.cfg)
        if getattr(self.cfg, 'use_custom_sampler', False):
            tr = tr.reset_index(drop=True)
        train_ds = claz(
            df=tr,
            transforms=self.cfg.transform['train'],
            cfg=self.cfg,
            phase='train',
            current_epoch=self.trainer.current_epoch,
        )
        if getattr(self.cfg, 'use_custom_sampler', False):
            return DataLoader(train_ds, batch_size=self.cfg.batch_size,
                num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn,
                sampler=InterleavedMaskClassBatchSampler(tr, self.cfg))

        else:
            return DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True,
                num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn)

    def get_val(self):
        val = self.cfg.train_df[self.cfg.train_df.fold == self.cfg.fold]
        if getattr(self.cfg, 'use_custom_sampler', False):
            val = val[val.mask_class==0]
        return val

    # Trainer.fit() 時に呼び出される
    def val_dataloader(self):
        val = self.get_val()

        print('len(valid):', len(val))
        claz = get_dataset_class(self.cfg)

        valid_ds = claz(
            df=val,
            transforms=self.cfg.transform['val'],
            cfg=self.cfg,
            phase='valid'
        )

        return DataLoader(valid_ds, batch_size=self.cfg.batch_size, shuffle=False, drop_last=False,
                          num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn)

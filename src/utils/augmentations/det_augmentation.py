import torch
import torchvision.transforms as T
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
from .keroppi import *

def det_noaug(size=None):
    return {
        'train': A.Compose([
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "val": A.Compose([
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            ToTensorV2(),
        ], p=1.0)}

def det_aug_base_noresize(size=None):
    return {
        'train': A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "val": A.Compose([
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            ToTensorV2(),
        ], p=1.0)}

def det_aug_base_v1(size):
    return {
        'train': A.Compose([
            A.Resize(height=size[0], width=size[1], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
            ],
            p=1.0,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
                label_fields=['labels']
            )
        ),
        "val": A.Compose([
            A.Resize(height=size[0], width=size[1], p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            A.Resize(height=size[0], width=size[1], p=1.0),
            ToTensorV2(),
        ], p=1.0)}



def det_aug_resize(size):
    if type(size) == int:
        size = (size, size)
    return {
        'train': A.Compose([
                A.Resize(height=size[0], width=size[1], p=1),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
                label_fields=['labels']
            )
        ),
        "val": A.Compose([
            A.Resize(size[0], size[1], p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            A.Resize(size[0], size[1], p=1.0),
            ToTensorV2(),
        ], p=1.0)}

def det_aug_base_v2(size):
    return {
        'train': A.Compose([
            A.Resize(size,size,always_apply=True),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.ShiftScaleRotate(),
            A.OneOf([
                A.RandomContrast(),
                A.RandomGamma(),
                A.RandomBrightness(),
                ], p=0.3),
            A.CLAHE(clip_limit=2),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "val": A.Compose([
            A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ], p=1.0)}

def det_aug_base_v3(size):
    return {
        'train': A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9,
                               border_mode=cv2.BORDER_CONSTANT),
            # A.OneOf([#off in most cases
                # A.OpticalDistortion(p=0.3),
            #     A.GridDistortion(p=.1),
            #     A.IAAPiecewiseAffine(p=0.3),
            # ], p=0.3),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "val": A.Compose([
            A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ], p=1.0)}

def det_aug_komugi_debug(size):
    print('size:', size)

    return {
        'train': A.Compose(
            [
                A.ToGray(p=0.01),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(height=size[0], width=size[1], p=1),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
                label_fields=['labels']
            )
        ),
        "val": A.Compose([
            A.Resize(height=size[0], width=size[1], p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            A.Resize(height=size[0], width=size[1], p=1.0),
            ToTensorV2(),
        ], p=1.0)}

def komugi_1st_aug(size):
    return {
        'train': A.Compose([
            A.Resize(height=size[0], width=size[1], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ToGray(p=0.01),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                # A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                # A.CLAHE(),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.25),
            A.HueSaturationValue(p=0.25),
            ToTensorV2(),
            ],
            p=1.0,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
                label_fields=['labels']
            )
        ),
        "val": A.Compose([
            A.Resize(height=size[0], width=size[1], p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            A.Resize(height=size[0], width=size[1], p=1.0),
            ToTensorV2(),
        ], p=1.0)}

def grayscale_augment(p, n):
    augs = A.OneOf([
        A.RandomGamma(),
        A.RandomContrast(),
        A.RandomBrightness(),
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0, rotate_limit=0),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.15, rotate_limit=0),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0, rotate_limit=30),
        A.GaussianBlur(),
        A.IAAAdditiveGaussianNoise()
    ], p=1)
    return A.Compose([augs] * n, p=p)

def grayscale_aug(size):
    return {
        'train': A.Compose([
            A.Resize(height=size[0], width=size[1], p=1.0),
            grayscale_augment(p=0.9, n=3),
            ToTensorV2(),
            ],
            p=1.0,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
                label_fields=['labels']
            )
        ),
        "val": A.Compose([
            A.Resize(height=size[0], width=size[1], p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            A.Resize(height=size[0], width=size[1], p=1.0),
            ToTensorV2(),
        ], p=1.0)}

# 小麦1st:
# mosaic & (mixup/cutmix) & 以下


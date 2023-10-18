import torch
import torchvision.transforms as T
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
# from RandAugment import RandAugment
import sys
import os
from .autoaug import AutoAugment

# mean = (0.485, 0.456, 0.406)  # RGB
# std = (0.229, 0.224, 0.225)  # RGB

def resize_only(size):
    return {
        'train': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

def base_aug_v1(size):
    return {
        'train': A.Compose([
            A.Resize(size[0], size[1]),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

def base_aug_v2(size):
    return {
        'train': A.Compose([
            A.Resize(size[0], size[1]),
            A.RandomRotate90(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

def seg_aug_heavy_v1(size):
    return {
        'train': A.Compose([
            A.Resize(size[0], size[1]),
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.GaussNoise(),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.5),
            A.OneOf([
                A.Sharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.5),
            A.HueSaturationValue(p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ]),
        "val": A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ],
        p=1.0),
    }

def bo_melanoma_1st(size):
    return {
        'train': A.Compose([
            A.Resize(size[0], size[1]),
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightness(limit=0.2, p=0.75),
            A.RandomContrast(limit=0.2, p=0.75),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=0.7),

            A.CLAHE(clip_limit=4.0, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            A.Resize(size, size),
            A.Cutout(max_h_size=int(size * 0.375), max_w_size=int(size * 0.375), num_holes=1, p=0.7),
            A.Normalize(),
            ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

def rsna_cervical_1st(size):
    return {
        'train': A.Compose([
            A.Resize(size[0], size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.Transpose(p=0.5),
            A.RandomBrightness(limit=0.1, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=45, border_mode=4, p=0.7),
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
                A.GaussNoise(var_limit=(3.0, 9.0)),
            ], p=0.5),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.),
                A.GridDistortion(num_steps=5, distort_limit=1.),
            ], p=0.5),
            A.Cutout(max_h_size=int(size[0] * 0.5), max_w_size=int(size[1] * 0.5), num_holes=1, p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

def medical_v1(size):
    return {
        'train': A.Compose([
             A.Resize(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.1), max_w_size=int(size[1] * 0.1), num_holes=5, p=0.5),
             A.Normalize(),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }
def medical_v1_no_norm(size):
    return {
        'train': A.Compose([
             A.Resize(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.1), max_w_size=int(size[1] * 0.1), num_holes=5, p=0.5),
             # A.Normalize(),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            # A.Normalize(),
            ToTensorV2(),
        ]),
    }
def medical_v1_no_norm_no_tensor(size):
    return {
        'train': A.Compose([
             A.Resize(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.1), max_w_size=int(size[1] * 0.1), num_holes=5, p=0.5),
             # A.Normalize(),
             # ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            # A.Normalize(),
            # ToTensorV2(),
        ]),
    }

def rsna4th_aug_v1(size):
    return {
        'train': A.ReplayCompose([
            A.LongestMaxSize(size[0]),
            A.PadIfNeeded(size[0], size[1], border_mode=cv2.BORDER_CONSTANT),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT,
                               p=0.5),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
        ]),
        'val': A.ReplayCompose([
            A.LongestMaxSize(size[0]),
            A.PadIfNeeded(size[0], size[1], border_mode=cv2.BORDER_CONSTANT),
        ]),
    }

def rsna4th_aug_v1(size):
    return {
        'train': A.ReplayCompose([
            A.LongestMaxSize(size[1]),
            A.PadIfNeeded(size[0], size[1], border_mode=cv2.BORDER_CONSTANT),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT,
                               p=0.5),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
        ]),
        'val': A.ReplayCompose([
            A.LongestMaxSize(size[1]),
            A.PadIfNeeded(size[0], size[1], border_mode=cv2.BORDER_CONSTANT),
        ]),
    }
def rsna4th_aug_v2(size):
    return {
        'train': A.ReplayCompose([
            A.LongestMaxSize(size[0]),
            A.PadIfNeeded(size[0], size[1], border_mode=cv2.BORDER_CONSTANT),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT,
                               p=0.5),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.OneOf([A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, distort_limit=0.1),
                     A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT)], p=0.2),
        ]),
        'val': A.ReplayCompose([
            A.LongestMaxSize(size[0]),
            A.PadIfNeeded(size[0], size[1], border_mode=cv2.BORDER_CONSTANT),
        ]),
    }

def replay_medical_v1_no_norm_no_tensor(size):
    return {
        'train': A.ReplayCompose([
             A.Resize(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.1), max_w_size=int(size[1] * 0.1), num_holes=5, p=0.5),
             # A.Normalize(),
             # ToTensorV2(),
        ]),
        'val': A.ReplayCompose([
            A.Resize(size[0], size[1]),
            # A.Normalize(),
            # ToTensorV2(),
        ]),
    }
def medical_v1_no_norm_no_tensor_with_vflip(size):
    return {
        'train': A.Compose([
             A.Resize(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.1), max_w_size=int(size[1] * 0.1), num_holes=5, p=0.5),
             # A.Normalize(),
             # ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            # A.Normalize(),
            # ToTensorV2(),
        ]),
    }
def medical_v1_no_norm_no_tensor_no_resize(size):
    return {
        'train': A.Compose([
             A.HorizontalFlip(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.1), max_w_size=int(size[1] * 0.1), num_holes=5, p=0.5),
             # A.Normalize(),
             # ToTensorV2(),
        ]),
        'val': A.Compose([
            # A.Resize(size[0], size[1]),
            # A.Normalize(),
            # ToTensorV2(),
        ]),
    }
def medical_v1_no_norm_no_tensor_multi_ch(size):
    return {
        'train': A.Compose([
             A.Resize(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             # A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.1), max_w_size=int(size[1] * 0.1), num_holes=5, p=0.5),
             # A.Normalize(),
             # ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            # A.Normalize(),
            # ToTensorV2(),
        ]),
    }
def medical_v1_1ch(size):
    return {
        'train': A.Compose([
             A.Resize(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.1), max_w_size=int(size[1] * 0.1), num_holes=5, p=0.5),
             A.Normalize(mean=(0.456), std=(0.224)),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(mean=(0.456), std=(0.224)),
            ToTensorV2(),
        ]),
    }

def medical_v1_4d(size):
    return {
        'train': A.Compose([
             A.Resize(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             A.Transpose(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.1), max_w_size=int(size[1] * 0.1), num_holes=5, p=0.5),
             A.Normalize(),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

def medical_v1_4d_cutout2(size):
    return {
        'train': A.Compose([
             A.Resize(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             A.Transpose(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.2), max_w_size=int(size[1] * 0.2), num_holes=5, p=0.5),
             A.Normalize(),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

def medical_v1_4d_cutout2_randomcrop_val_resize(size):
    return {
        'train': A.Compose([
             A.Resize(int(size[0]*1.2), int(size[1]*1.2)),
             A.RandomCrop(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             A.Transpose(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.2), max_w_size=int(size[1] * 0.2), num_holes=5, p=0.5),
             A.Normalize(),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

def medical_v1_4d_1ch(size):
    return {
        'train': A.Compose([
             A.Resize(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             A.Transpose(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.1), max_w_size=int(size[1] * 0.1), num_holes=5, p=0.5),
             A.Normalize(mean=(0.456), std=(0.224)),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(mean=(0.456), std=(0.224)),
            ToTensorV2(),
        ]),
    }

def medical_v1_4d_1ch_rm_transpose(size):
    return {
        'train': A.Compose([
             A.Resize(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.1), max_w_size=int(size[1] * 0.1), num_holes=5, p=0.5),
             A.Normalize(mean=(0.456), std=(0.224)),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(mean=(0.456), std=(0.224)),
            ToTensorV2(),
        ]),
    }

def medical_v1_4d_cutout2_1ch(size):
    return {
        'train': A.Compose([
             A.Resize(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             A.Transpose(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.2), max_w_size=int(size[1] * 0.2), num_holes=5, p=0.5),
             A.Normalize(mean=(0.456), std=(0.224)),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(mean=(0.456), std=(0.224)),
            ToTensorV2(),
        ]),
    }

def medical_v1_4d_cutout2_1ch_rm_transpose(size):
    return {
        'train': A.Compose([
             A.Resize(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             # A.Transpose(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.2), max_w_size=int(size[1] * 0.2), num_holes=5, p=0.5),
             A.Normalize(mean=(0.456), std=(0.224)),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(mean=(0.456), std=(0.224)),
            ToTensorV2(),
        ]),
    }

def medical_v1_4d_cutout2_rm_transpose(size):
    return {
        'train': A.Compose([
             A.Resize(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             # A.Transpose(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.2), max_w_size=int(size[1] * 0.2), num_holes=5, p=0.5),
             A.Normalize(),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

def medical_v1_4d_cutout2_rm_Affine(size):
    return {
        'train': A.Compose([
             A.Resize(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             # A.Transpose(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             # A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.2), max_w_size=int(size[1] * 0.2), num_holes=5, p=0.5),
             A.Normalize(),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

def kuma_aug_multi_image(size):
    return {
        'train': A.Compose([
            A.Resize(size[0], size[1]),
            A.ShiftScaleRotate(0.1, 0.2, 15),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.25),
            A.Normalize(),
            A.CoarseDropout(max_holes=20, max_height=64, max_width=64, p=0.2),
            ToTensorV2()
        ], additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image'}),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ], additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image'}),
    }

def kuma_aug(size):
    return {
        'train': A.Compose([
            A.Resize(size[0], size[1]),
            A.ShiftScaleRotate(0.1, 0.2, 15),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.25),
            A.Normalize(),
            A.CoarseDropout(max_holes=20, max_height=64, max_width=64, p=0.2),
            ToTensorV2()
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }


def medical_v1_4d_cutout2_randomcrop_val_centercrop(size):
    return {
        'train': A.Compose([
             A.Resize(int(size[0]*1.2), int(size[1]*1.2)),
             A.RandomCrop(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             A.Transpose(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.2), max_w_size=int(size[1] * 0.2), num_holes=5, p=0.5),
             A.Normalize(),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(int(size[0]*1.2), int(size[1]*1.2)),
            A.CenterCrop(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }


def medical_v1_4d_cutout3_randomcrop_val_centercrop_1ch(size):
    return {
        'train': A.Compose([
             A.Resize(int(size[0]*1.2), int(size[1]*1.2)),
             A.RandomCrop(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             A.Transpose(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.3), max_w_size=int(size[1] * 0.3), num_holes=5, p=0.5),
             A.Normalize(mean=(0.456), std=(0.224)),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(int(size[0]*1.2), int(size[1]*1.2)),
            A.CenterCrop(size[0], size[1]),
            A.Normalize(mean=(0.456), std=(0.224)),
            ToTensorV2(),
        ]),
    }

def medical_v1_4d_cutout2_randomcrop_val_centercrop_1ch(size):
    return {
        'train': A.Compose([
             A.Resize(int(size[0]*1.2), int(size[1]*1.2)),
             A.RandomCrop(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             A.Transpose(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.2), max_w_size=int(size[1] * 0.2), num_holes=5, p=0.5),
             A.Normalize(mean=(0.456), std=(0.224)),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(int(size[0]*1.2), int(size[1]*1.2)),
            A.CenterCrop(size[0], size[1]),
            A.Normalize(mean=(0.456), std=(0.224)),
            ToTensorV2(),
        ]),
    }


def medical_v1_4d_cutout2_randomcrop_val_centercrop_1ch_rm_transpose(size):
    return {
        'train': A.Compose([
             A.Resize(int(size[0]*1.2), int(size[1]*1.2)),
             A.RandomCrop(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             # A.Transpose(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.2), max_w_size=int(size[1] * 0.2), num_holes=5, p=0.5),
             A.Normalize(mean=(0.456), std=(0.224)),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(int(size[0]*1.2), int(size[1]*1.2)),
            A.CenterCrop(size[0], size[1]),
            A.Normalize(mean=(0.456), std=(0.224)),
            ToTensorV2(),
        ]),
    }

def bit8_medical_v1_4d_cutout2_randomcrop_val_centercrop_1ch_rm_transpose(size):
    return {
        'train': A.Compose([
             A.ToFloat(),
             A.Resize(int(size[0]*1.2), int(size[1]*1.2)),
             A.RandomCrop(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             # A.Transpose(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             # A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(blur_limit=5),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.2), max_w_size=int(size[1] * 0.2), num_holes=5, p=0.5),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.ToFloat(),
            A.Resize(int(size[0]*1.2), int(size[1]*1.2)),
            A.CenterCrop(size[0], size[1]),
            ToTensorV2(),
        ]),
    }

def bit16_medical_v1_4d_cutout2_randomcrop_val_centercrop_1ch_rm_transpose(size):
    return {
        'train': A.Compose([
             A.ToFloat(max_value=65535.0),
             A.Resize(int(size[0]*1.2), int(size[1]*1.2)),
             A.RandomCrop(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             # A.Transpose(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             # A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(blur_limit=5),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.2), max_w_size=int(size[1] * 0.2), num_holes=5, p=0.5),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.ToFloat(max_value=65535.0),
            A.Resize(int(size[0]*1.2), int(size[1]*1.2)),
            A.CenterCrop(size[0], size[1]),
            ToTensorV2(),
        ]),
    }


def medical_v1_4d_cutout2_randomcrop_v3_val_centercrop_1ch_rm_transpose(size):
    return {
        'train': A.Compose([
             A.Resize(int(size[0]*1.1), int(size[1]*1.1)),
             A.RandomCrop(size[0], size[1]),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             # A.Transpose(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.2), max_w_size=int(size[1] * 0.2), num_holes=5, p=0.5),
             A.Normalize(mean=(0.456), std=(0.224)),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(int(size[0]*1.1), int(size[1]*1.1)),
            A.CenterCrop(size[0], size[1]),
            A.Normalize(mean=(0.456), std=(0.224)),
            ToTensorV2(),
        ]),
    }


def guie_aug(size):
    return {
        'train': T.Compose([
             # T.Resize(size[0]),
             # A.HorizontalFlip(p=0.5),
             # A.VerticalFlip(),
             # A.ShiftScaleRotate(p=0.5),
             # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             # A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             # A.CLAHE(clip_limit=(1,4), p=0.5),
             # A.OneOf([
             #     A.OpticalDistortion(distort_limit=1.0),
             #     A.GridDistortion(num_steps=5, distort_limit=1.),
             #     A.ElasticTransform(alpha=3),
             # ], p=0.2),
             # A.OneOf([
             #     A.GaussNoise(var_limit=[10, 50]),
             #     A.GaussianBlur(),
             #     A.MotionBlur(),
             #     A.MedianBlur(),
             # ], p=0.2),
             # A.PiecewiseAffine(p=0.2),
             # A.Sharpen(p=0.2),
             # A.Cutout(max_h_size=int(size[0] * 0.1), max_w_size=int(size[1] * 0.1), num_holes=5, p=0.5),
            # A.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            # ),
            #  ToTensorV2(),
            T.PILToTensor()
        ]),
        'val': T.Compose([
             # T.Resize(size[0]),
            T.PILToTensor()
        ]),
    }

def for_nfl(size):
    return {
        'train': A.Compose([
             A.RandomResizedCrop(size[0], size[1], scale=(0.9, 1), p=1),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.Resize(size[0], size[1]),
             A.OneOf([
                 A.JpegCompression(),
                 A.Downscale(scale_min=0.1, scale_max=0.15),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             A.Cutout(max_h_size=int(size[0] * 0.1), max_w_size=int(size[1] * 0.1), num_holes=5, p=0.5),
             A.Normalize(),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

def ian_grayscale_augment(p=0.8, n=3):
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
    return A.Compose([augs]*n, p=p)

def grayscale_augment(p, n):
    return {
        'train': A.Compose([
            A.Resize(size[0], size[1]),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            ian_grayscale_augment(p=0.8, n=3),
            A.Normalize(),
            ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

def bo_melanoma_1st(size):
    return {
        'train': A.Compose([
            A.Resize(size[0], size[1]),
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightness(limit=0.2, p=0.75),
            A.RandomContrast(limit=0.2, p=0.75),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=0.7),

            A.CLAHE(clip_limit=4.0, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            A.Cutout(max_h_size=int(size[0] * 0.375), max_w_size=int(size[1] * 0.375), num_holes=1, p=0.7),
            A.Normalize(),
            ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

def bo_melanoma_1st_no_transpose(size):
    return {
        'train': A.Compose([
            A.Resize(size[0], size[1]),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightness(limit=0.2, p=0.75),
            A.RandomContrast(limit=0.2, p=0.75),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=0.7),

            A.CLAHE(clip_limit=4.0, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            A.Cutout(max_h_size=int(size[0] * 0.375), max_w_size=int(size[1] * 0.375), num_holes=1, p=0.7),
            A.Normalize(),
            ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

def bo_melanoma_1st_rand(size):
    return {
        'train': A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightness(limit=0.2, p=0.75),
            A.RandomContrast(limit=0.2, p=0.75),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=0.7),

            A.CLAHE(clip_limit=4.0, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            RandAugment(5, 30),
            A.Resize(size[0], size[1]),
            A.Cutout(max_h_size=int(size[0] * 0.375), max_w_size=int(size[1] * 0.375), num_holes=1, p=0.7),
            A.Normalize(),
            ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

def autoaug(size):
    return {
        'train': A.Compose([
            A.Resize(size[0], size[1]),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            AutoAugment(),
            A.Normalize(),
            ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

def base_aug_v1_4ch(size):
    mean = (0.485, 0.456, 0.406, 0.406)  # RGBY
    std = (0.229, 0.224, 0.225, 0.225)  # RGBY
    return {
        'train': A.Compose([
            A.Resize(size[0], size[1]),
            T.RandomHorizontalFlip(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ]),
    }

def base_aug_4ch_random_crop(size):
    mean = (0.485, 0.456, 0.406, 0.406)  # RGBY
    std = (0.229, 0.224, 0.225, 0.225)  # RGBY
    return {
        'train': A.Compose([
            A.Resize(int(size[0]*1.5), int(size[1]*1.5)),
            A.RandomCrop(size[0], size[1]),
            T.RandomHorizontalFlip(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ]),
        'val': A.Compose([
            A.Resize(int(size[0]*1.5), int(size[1]*1.5)),
            A.RandomCrop(size[0], size[1]),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ]),
    }


def debug_aug(size):
    return {'train': A.Compose([
                # A.HorizontalFlip(p=0.5),
                A.ImageCompression(quality_lower=99, quality_upper=100),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
                A.Resize(size[0], size[1]),
                A.Cutout(max_h_size=int(size[0] * 0.4), max_w_size=int(size[1] * 0.4), num_holes=1, p=0.5),
                A.Normalize(
                    mean=[0.40188728, 0.44177499, 0.48930021],
                    std=[0.11740651, 0.10739016, 0.10927369],
                ),
                ToTensorV2(),
            ]),
            'val': A.Compose([
                A.Resize(size[0], size[1]),
                A.Normalize(
                    mean=[0.40188728, 0.44177499, 0.48930021],
                    std=[0.11740651, 0.10739016, 0.10927369],
                ),
                ToTensorV2(),
            ])}

# import kornia
# class DataAugmentation1(nn.Module):
#     def __init__(self,):
#         super().__init__()
#         self.flip = nn.Sequential(
#             kornia.augmentation.RandomHorizontalFlip(p=0.5),
#             kornia.augmentation.RandomVerticalFlip(p=0.5),
#         )

#         p=0.8
#         self.transform_geometry = ImageSequential(
#             kornia.augmentation.RandomAffine(degrees=20, translate=0.1, scale=[0.8,1.2], shear=20, p=p),
#             kornia.augmentation.RandomThinPlateSpline(scale=0.25, p=p),
#             random_apply=1, #choose 1
#         )

#         p=0.5
#         self.transform_intensity = ImageSequential(
#             kornia.augmentation.RandomGamma(gamma=(0.5, 1.5), gain=(0.5, 1.2), p=p),
#             kornia.augmentation.RandomContrast(contrast=(0.8,1.2), p=p),
#             kornia.augmentation.RandomBrightness(brightness=(0.8,1.2), p=p),
#             random_apply=1, #choose 1
#         )

#         p=0.5
#         self.transform_other = ImageSequential(
#             kornia.augmentation.MyRoll(p=0.1), #Mosaic Augmentation using only one image, implemented by using pytorch roll , i.e. cyclic shift
#             kornia.augmentation.MyCutOut(num_block=5, block_size=[0.1, 0.2], fill='constant', p=0.1),
#             random_apply=1, #choose 1
#         )


#     @torch.no_grad()  # disable gradients for effiency
#     def forward(self, x):
#         x = self.flip(x)  # BxCxHxW
#         x = self.transform_geometry(x)
#         x = self.transform_intensity(x)
#         x = self.transform_other(x)
#         return x

def nodoca_aug(size):
    return {
            'train': A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.4, rotate_limit=180, p=1,
                                   border_mode=cv2.BORDER_REFLECT_101),
                A.OneOf([#off in most cases
                    A.MotionBlur(blur_limit=3, p=0.1),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.OneOf(
                    [  # off in most cases
                        A.OpticalDistortion(p=1),
                        A.GridDistortion(p=1),
                        # A.IAAPerspective(p=1),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.HueSaturationValue(10, 15, 10),
                        A.RandomBrightnessContrast(),
                        A.RandomFog(),
                    ],
                    p=0.3,
                ),
                A.Resize(size[0], size[1]),
                ToTensorV2(),
            ], additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image'}, p=1),
        "val": A.Compose([A.Resize(size[0], size[1]), ToTensorV2()], additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image'}),
    }
def nodoca_aug_wo_tensor(size):
    return {
            'train': A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.4, rotate_limit=180, p=1,
                                   border_mode=cv2.BORDER_REFLECT_101),
                A.OneOf([#off in most cases
                    A.MotionBlur(blur_limit=3, p=0.1),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.OneOf(
                    [  # off in most cases
                        A.OpticalDistortion(p=1),
                        A.GridDistortion(p=1),
                        # A.IAAPerspective(p=1),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.HueSaturationValue(10, 15, 10),
                        A.RandomBrightnessContrast(),
                        A.RandomFog(),
                    ],
                    p=0.3,
                ),
                A.Resize(size[0], size[1]),
            ], p=1),
        "val": A.Compose([A.Resize(size[0], size[1])]),
    }
def nodoca_aug_wo_tensor_wo_resize(size):
    return {
            'train': A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.4, rotate_limit=180, p=1,
                                   border_mode=cv2.BORDER_REFLECT_101),
                A.OneOf([#off in most cases
                    A.MotionBlur(blur_limit=3, p=0.1),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.OneOf(
                    [  # off in most cases
                        A.OpticalDistortion(p=1),
                        A.GridDistortion(p=1),
                        # A.IAAPerspective(p=1),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.HueSaturationValue(10, 15, 10),
                        A.RandomBrightnessContrast(),
                        A.RandomFog(),
                    ],
                    p=0.3,
                ),
                # A.Resize(size[0], size[1]),
            ], p=1),
        "val": A.Compose([]),
    }

def nodoca_aug_early_resize(size):
    return {
            'train': A.Compose([
                A.Resize(size[0], size[1]),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.4, rotate_limit=180, p=1,
                                   border_mode=cv2.BORDER_REFLECT_101),
                A.OneOf([#off in most cases
                    A.MotionBlur(blur_limit=3, p=0.1),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.OneOf(
                    [  # off in most cases
                        A.OpticalDistortion(p=1),
                        A.GridDistortion(p=1),
                        # A.IAAPerspective(p=1),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.HueSaturationValue(10, 15, 10),
                        A.RandomBrightnessContrast(),
                        A.RandomFog(),
                    ],
                    p=0.3,
                ),
                ToTensorV2(),
            ], p=1),
        "val": A.Compose([A.Resize(size[0], size[1]), ToTensorV2()]),
    }

def nodoca_aug_with_norm(size):
    return {
            'train': A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.4, rotate_limit=180, p=1,
                                   border_mode=cv2.BORDER_REFLECT_101),
                A.OneOf([#off in most cases
                    A.MotionBlur(blur_limit=3, p=0.1),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.OneOf(
                    [  # off in most cases
                        A.OpticalDistortion(p=1),
                        A.GridDistortion(p=1),
                        # A.IAAPerspective(p=1),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.HueSaturationValue(10, 15, 10),
                        A.RandomBrightnessContrast(),
                        A.RandomFog(),
                    ],
                    p=0.3,
                ),
                A.Resize(size[0], size[1]),
                A.Normalize(),
                ToTensorV2(),
            ], p=1),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ])
    }

def rsna_1st_aug(size):
    return {
            'train': A.Compose([
                A.Resize(size[0], size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.RandomBrightness(limit=0.1, p=0.7),
                A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=45, border_mode=4, p=0.7),

                A.OneOf([
                    A.MotionBlur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                    A.GaussNoise(var_limit=(3.0, 9.0)),
                ], p=0.5),
                A.OneOf([
                    A.OpticalDistortion(distort_limit=1.),
                    A.GridDistortion(num_steps=5, distort_limit=1.),
                ], p=0.5),

                A.Cutout(max_h_size=int(size[0] * 0.5), max_w_size=int(size[1] * 0.5), num_holes=1, p=0.5),
            ], p=1),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
        ])
    }


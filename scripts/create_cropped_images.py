import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import copy
from monai.transforms import Resize
from scipy.special import softmax
from multiprocessing import Pool
import warnings
import os
warnings.simplefilter('ignore')

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def aggregate_3dseg_results(arg):
    # The predictions of the 10 models from resnet18 and resnet50 are averaged, and sigmoid, and saved as np.uint8 mask with 0.2 as the threshold value.
    image_id, itr = arg
    pred_masks = [np.load(f'../results/rsna_3d_seg_resnet18/test_preds_fold{fold}/{image_id}.npy') for fold in range(5)]
    pred_masks += [np.load(f'../results/rsna_3d_seg_resnet50/test_preds_fold{fold}/{image_id}.npy') for fold in range(5)]
    pred_masks = np.mean([pred_masks, pred_masks2], 0)
    pred_masks = sigmoid(pred_masks)
    pred_masks = (pred_masks>0.2).astype(np.uint8)
    np.save(f'../results/resnet2models/test_preds_sigmoid/{image_id}.npy', pred_masks)


def crop_based_on_mask(ct_img, mask, pad_x, pad_y, pad_z):
    idx = np.where(mask >= 1)

    min_idx = np.min(idx, axis=1)
    max_idx = np.max(idx, axis=1) + 1

    min_z = max(0, min_idx[0]-pad_z)
    min_y = max(0, min_idx[1]-pad_y)
    min_x = max(0, min_idx[2]-pad_x)
    max_z = min(max_idx[0]+pad_z, ct_img.shape[0])
    max_y = min(max_idx[1]+pad_y, ct_img.shape[1])
    max_x = min(max_idx[2]+pad_x, ct_img.shape[2])

    cropped_img = ct_img[min_z:max_z, min_y:max_y, min_x:max_x]

    return cropped_img, min_z, max_z, min_y, max_y, min_x, max_x
def crop_based_on_mask_masked(ct_img_origin, mask, pad_x, pad_y, pad_z):
    ct_img = ct_img_origin.copy()
    ct_img = ct_img * mask
    idx = np.where(mask >= 1)

    min_idx = np.min(idx, axis=1)
    max_idx = np.max(idx, axis=1) + 1
    min_z = max(0, min_idx[0]-pad_z)
    min_y = max(0, min_idx[1]-pad_y)
    min_x = max(0, min_idx[2]-pad_x)
    max_z = min(max_idx[0]+pad_z, ct_img.shape[0])
    max_y = min(max_idx[1]+pad_y, ct_img.shape[1])
    max_x = min(max_idx[2]+pad_x, ct_img.shape[2])

    cropped_img = ct_img[min_z:max_z, min_y:max_y, min_x:max_x]

    return cropped_img, min_z, max_z, min_y, max_y, min_x, max_x

def crop_and_save(arg):
    image_id, itr = arg
    ims = np.load(itr.path.values[0])
    if skip_each_n_slice != 1:
        ims = ims[[i for i in range(len(ims)) if i % skip_each_n_slice == 0]]
    if os.path.exists(f'../results/resnet2models/test_preds_sigmoid/{image_id}.npy'):
        pred_masks = np.load(f'../results/resnet2models/test_preds_sigmoid/{image_id}.npy')
    else:
        pred_masks = np.load(f'../results/resnet2models/test_preds_sigmoid/{image_id.split("_")[1]}.npy')

    resize = Resize(ims.shape[:3])
    pred_masks = resize(pred_masks)
    all_mask = pred_masks[0]+pred_masks[1]+pred_masks[2]+pred_masks[3]+pred_masks[4]

    idx = np.where(all_mask >= 1)
    min_idx = np.min(idx, axis=1)
    max_idx = np.max(idx, axis=1) + 1
    if pad_ratio == 0:
        pad_x = 0
        pad_y = 0
        pad_z = 0
    else:
        pad_x = round((max_idx[2]-min_idx[2])/pad_ratio)
        pad_y = round((max_idx[1]-min_idx[1])/pad_ratio)
        pad_z = round((max_idx[0]-min_idx[0])/pad_z_ratio)

    dfs = []
    origin_path = itr.path.values[0].replace(image_dir,  f'resnet2models_0.2_{pad_ratio}_{pad_z_ratio}')
    for class_ in range(5):
        if np.sum(pred_masks[class_])>0:
            pred_ims, min_z, max_z, min_y, max_y, min_x, max_x = crop_based_on_mask(ims, pred_masks[class_], pad_x, pad_y, pad_z)
            new_path = origin_path.replace('.npy', f'_{class_}.npy')
            np.save(new_path, pred_ims)
            citr = copy.deepcopy(itr)
            citr['path'] = new_path
            citr['slice_len'] = len(pred_ims)
            citr['image_height'] = pred_ims.shape[1]
            citr['image_width'] = pred_ims.shape[2]
            citr['mask_class'] = class_
            citr['x_min'] = min_x
            citr['x_max'] = max_x
            citr['y_min'] = min_y
            citr['y_max'] = max_y
            citr['z_min'] = min_z
            citr['z_max'] = max_z
            dfs.append(citr)

    # class 2&3
    if np.sum(pred_masks[2]+pred_masks[3])>0:
        pred_ims, min_z, max_z, min_y, max_y, min_x, max_x = crop_based_on_mask(ims, pred_masks[2]+pred_masks[3], pad_x, pad_y, pad_z)
        new_path = origin_path.replace('.npy', f'_23.npy')
        np.save(new_path, pred_ims)
        citr = itr.copy()
        citr['path'] = new_path
        citr['slice_len'] = len(pred_ims)
        citr['image_height'] = pred_ims.shape[1]
        citr['image_width'] = pred_ims.shape[2]
        citr['mask_class'] = 34
        citr['x_min'] = min_x
        citr['x_max'] = max_x
        citr['y_min'] = min_y
        citr['y_max'] = max_y
        citr['z_min'] = min_z
        citr['z_max'] = max_z
        dfs.append(citr)

    return dfs

def crop_and_masking_save(arg):
    image_id, itr = arg
    ims = np.load(itr.path.values[0])
    if skip_each_n_slice != 1:
        ims = ims[[i for i in range(len(ims)) if i % skip_each_n_slice == 0]]


    if os.path.exists(f'../results/resnet2models/test_preds_sigmoid/{image_id}.npy'):
        pred_masks = np.load(f'../results/resnet2models/test_preds_sigmoid/{image_id}.npy')
    else:
        pred_masks = np.load(f'../results/resnet2models/test_preds_sigmoid/{image_id.split("_")[1]}.npy')

    resize = Resize(ims.shape[:3])
    pred_masks = resize(pred_masks)

    all_mask = pred_masks[0]+pred_masks[1]+pred_masks[2]+pred_masks[3]+pred_masks[4]

    idx = np.where(all_mask >= 1)
    min_idx = np.min(idx, axis=1)
    max_idx = np.max(idx, axis=1) + 1

    if pad_ratio == 0:
        pad_x = 0
        pad_y = 0
        pad_z = 0
    else:
        pad_x = round((max_idx[2]-min_idx[2])/pad_ratio)
        pad_y = round((max_idx[1]-min_idx[1])/pad_ratio)
        pad_z = round((max_idx[0]-min_idx[0])/pad_z_ratio)
    dfs = []
    origin_path = itr.path.values[0].replace(image_dir,  'resnet2models_0.2_0_0_masked')
    class_ = 0
    if np.sum(pred_masks[class_])>0:
        pred_ims, min_z, max_z, min_y, max_y, min_x, max_x = crop_based_on_mask_masked(ims, pred_masks[class_], pad_x, pad_y, pad_z)
        new_path = origin_path.replace('.npy', f'_{class_}.npy')
#             print(new_path)
        np.save(new_path, pred_ims)
        citr = copy.deepcopy(itr)
        citr['path'] = new_path
        citr['slice_len'] = len(pred_ims)
        citr['image_height'] = pred_ims.shape[1]
        citr['image_width'] = pred_ims.shape[2]
        citr['mask_class'] = class_
        citr['x_min'] = min_x
        citr['x_max'] = max_x
        citr['y_min'] = min_y
        citr['y_max'] = max_y
        citr['z_min'] = min_z
        citr['z_max'] = max_z
        dfs.append(citr)
    return dfs


if __name__ == "__main__":

    test = pd.read_csv('../input/test_for_3dseg.csv')
    os.system(f'mkdir -p ../input/segmentation_3d/resnet2models_0.2_30_25')
    os.system(f'mkdir -p ../input/segmentation_3d/resnet2models_0.2_8_8')
    os.system(f'mkdir -p ../input/segmentation_3d/resnet2models_0.2_0_0_masked')
    os.system(f'mkdir -p ../results/resnet2models/test_preds_sigmoid')

    ### aggregate_3dseg_results ###
    # The predictions of the 10 models from resnet18 and resnet50 are averaged, and sigmoid, and saved as np.uint8 mask with 0.2 as the threshold value.
    p = Pool(processes=4)
    df_list = list(test.groupby('image_id'))
    with tqdm(total=len(df_list)) as pbar:
        for _ in p.imap(aggregate_3dseg_results, df_list):
            pbar.update(1)
    p.close()

    pad_ratio = 30
    pad_z_ratio = 25
    p = Pool(processes=4)
    dfs = []
    df_list = list(test.groupby('image_id'))
    with tqdm(total=len(df_list)) as pbar:
        for this_df in p.imap(crop_and_save, df_list):
            dfs += this_df
            pbar.update(1)
    p.close()

    d = pd.concat(dfs).to_csv(path, index=False)
    tr = pd.read_csv('../input/train_with_path.csv')

    tr['image_id'] = tr['patient_id'].astype(str) + '_' + tr['series_id'].astype(str)
    tr = tr.drop_duplicates('image_id')
    del tr['path']
    d = d.merge(tr, on='image_id')
    path = f'../input/train2.5d_2models_th0.2_pad_{pad_ratio}_{pad_z_ratio}.csv'
    d.to_csv(path, index=False)
    print('save to', path)

    pad_ratio = 8
    pad_z_ratio = 8
    p = Pool(processes=4)
    dfs = []
    df_list = list(test.groupby('image_id'))
    with tqdm(total=len(df_list)) as pbar:
        for this_df in p.imap(crop_and_save, df_list):
            dfs += this_df
            pbar.update(1)
    p.close()

    d = pd.concat(dfs).to_csv(path, index=False)
    tr = pd.read_csv('../input/train_with_path.csv')

    tr['image_id'] = tr['patient_id'].astype(str) + '_' + tr['series_id'].astype(str)
    tr = tr.drop_duplicates('image_id')
    del tr['path']
    d = d.merge(tr, on='image_id')
    path = f'../input/train2.5d_2models_th0.2_pad_{pad_ratio}_{pad_z_ratio}.csv'
    d.to_csv(path, index=False)
    print('save to', path)

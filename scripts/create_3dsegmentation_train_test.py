import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import copy
from sklearn.model_selection import StratifiedKFold

def decompose_mask(mask, num_classes=6):
    new_mask = np.empty((num_classes-1, mask.shape[0], mask.shape[1]))

    for i in range(1, num_classes):
        new_mask[i-1] = (mask == i)
    return new_mask

def add_stratified_fold(df, col, cut=False):
    if cut:
        df[f'{col}_cut'] = pd.cut(df[col], 10, labels=False)
        col = f'{col}_cut'

    df = df.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 5)
    df['fold'] = -1
    for n, (trn_idx, val_idx) in enumerate(skf.split(df, df[col])):
        df.loc[val_idx, 'fold'] = n
    if cut:
        del df[col]
    return df


### create train ###
df = pd.read_csv('../input/train_with_path.csv')

for mask_path in glob('../input/segmentations/*'):
    mask = nib.load(mask_path).get_fdata()
    series_id = int(mask_path.split('/')[-1])
    idf = df[df.series_id==series_id]
    print(idf.patient_id.nunique(), series_id)

    ims = []
    masks = []

    for path_n, path in tqdm(enumerate(idf.path)):
        m = mask[:, :, path_n].T
        m = m[::-1, :]
        m = decompose_mask(m)
        ims.append(im)
        masks.append(m)
    ims = np.array(ims)
    masks = np.array(masks).transpose((1,0,2,3))
    assert masks.shape[1] == ims.shape[0]
    np.save(f'..input/segmentation_3d/image/{series_id}.npy', ims)
    np.save(f'../input/segmentation_3d/mask/{series_id}.npy', masks)

ids = []
lens=[]
for mask_path in glob('../input/segmentations/*'):
    series_id = int(mask_path.split('/')[-1])
    ids.append(series_id)
    lens.append(len(df[df.series_id==series_id]))

mdf = pd.DataFrame({'image_id': ids, 'slice_len': lens})
mdf['path'] = '../input/segmentation_3d/image/' + mdf.image_id.astype(str) + '.npy'
mdf['mask_path'] = '../input/segmentation_3d/mask/' + mdf.image_id.astype(str) + '.npy'
mdf = add_stratified_fold(mdf, 'slice_len', cut=True)
mdf.to_csv(f'../input/train_for_3dseg.csv', index=False)
### create train ###


### create test ###
df = pd.read_csv('../input/train_with_path.csv')
paths = []
for id, idf in tqdm(df.groupby(['patient_id', 'series_id'])):
    ims = []
    for path in idf.path:
        im = cv2.imread(path)[:,:,0]
        ims.append(im)
    ims = np.array(ims)
    new_path = f'../input/segmentation_3d/image/{"_".join(list(map(str,id)))}.npy'
    np.save(new_path, ims)
    paths.append(new_path)
pd.DataFrame({'path': paths}).to_csv('../input/test_for_3dseg.csv', index=False)
### create test ###

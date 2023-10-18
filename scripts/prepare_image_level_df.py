import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import copy
import warnings
warnings.simplefilter('ignore')
from scipy.special import softmax
def sigmoid(x):
    return 1/(1 + np.exp(-x))

if __name__ == "__main__":
    im = pd.read_csv('../input/image_level_labels.csv')
    im['slice'] = im.instance_number.apply(lambda x: str(x).zfill(4))
    im['image_id'] = im['patient_id'].astype(str) + '_' + im['series_id'].astype(str)
    im['InstanceNumber'] = im['instance_number'].values
    im['image_bowel_injury'] = 0
    im['image_extravasation_injury'] = 0
    im.loc[im.injury_name=='Bowel', 'image_bowel_injury'] = 1
    im.loc[im.injury_name=='Active_Extravasation', 'image_extravasation_injury'] = 1
    tmp = im.groupby('path')[['image_extravasation_injury', 'image_bowel_injury']].sum().reset_index()
    del im['image_bowel_injury'], im['image_extravasation_injury']
    im = im.drop_duplicates('path').merge(tmp, on='path')

    meta = pd.read_parquet('../input/train_dicom_tags.parquet')
    meta['patient_id'] = meta.PatientID.values
    meta['series_id'] = meta.SeriesInstanceUID.apply(lambda x: x.split('.')[-1])
    meta['image_id'] = meta['patient_id'].astype(str) + '_' + meta['series_id'].astype(str)
    meta['z'] = meta.ImagePositionPatient.apply(lambda x: eval(x)[-1])
    meta = meta.sort_values(['image_id', 'z'])
    meta = meta.merge(im[['InstanceNumber', 'image_id', 'injury_name', 'image_bowel_injury', 'image_extravasation_injury']], on=['image_id','InstanceNumber'], how='left')

    meta['image_bowel_injury'] = meta.image_bowel_injury.fillna(0)
    meta['image_extravasation_injury'] = meta.image_extravasation_injury.fillna(0)

    df = pd.read_csv('../input/train_with_path.csv')
    df['image_id'] = df['patient_id'].astype(str) + '_' + df['series_id'].astype(str)

    neg = df[(df.extravasation_injury==0) & (df.bowel_injury==0)]
    neg['image_bowel_injury'] = 0
    neg['image_extravasation_injury'] = 0
    pos = df[(df.extravasation_injury==1) | (df.bowel_injury==1)]
    pos = pos[pos.image_id.isin(im.image_id)]
    dfs = []
    for id, idf in tqdm(pos.groupby('image_id')):
        mdf=meta[meta.image_id==id]
        idf = df[df.image_id==id]
        assert len(idf) == len(mdf)
        idf[['InstanceNumber', 'z', 'image_bowel_injury', 'image_extravasation_injury']] = mdf[['InstanceNumber', 'z', 'image_bowel_injury', 'image_extravasation_injury']].values
        dfs.append(idf)
    pos = pd.concat(dfs)

    dfs=[]
    for i, idf in pos.groupby('image_id'):
        if (idf['extravasation_injury'].sum()>0) & (idf['image_extravasation_injury'].sum()==0):
            continue
        dfs.append(idf)
    pos = pd.concat(dfs)
    al = pd.concat([pos, neg])
    al.fold.value_counts()

    al.to_csv('../input/train_2classes_for_pretrain.csv', index=False)

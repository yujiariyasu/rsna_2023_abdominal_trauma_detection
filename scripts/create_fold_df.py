import os
import numpy as np
import pandas as pd

def prepare_folds(df, k=4):
    cols = [
        'series_count','bowel_injury', 'extravasation_injury', 'kidney_low',
        'kidney_high', 'liver_low', 'liver_high', 'spleen_low', 'spleen_high'
    ]
    df = df.reset_index(drop=True)
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    splits = mskf.split(df, y=df[cols])

    df['fold'] = -1
    for i, (_, val_idx) in enumerate(splits):
        df.loc[val_idx, "fold"] = i
    return df

if __name__ == "__main__":
    df = pd.read_csv('../input/train2.5d_2models_th0.2_pad_8_8.csv')
    df = df.drop_duplicates('image_id')
    tmp = df.groupby('patient_id')['image_id'].count().reset_index()
    tmp.columns=['patient_id', 'series_count']
    df = df.merge(tmp, on='patient_id')

    folds = prepare_folds(df.drop_duplicates('patient_id'), k = 4)
    folds[['patient_id', 'fold']].to_csv('../input/multi_label_stratified_folds.csv', index=False)

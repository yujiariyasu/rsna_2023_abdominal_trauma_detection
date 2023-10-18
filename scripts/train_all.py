import os
configs = [
    'class0_gru_chaug_256_2segmodels_v3',
    'class0_masked_192_2segmodels_pad0',
    'class0_masked_192_2segmodels_pad0_caformer',
    'class23_2segmodels_v3_xcit_small',
    'class1_lstm_112_2segmodels_20epochs_auc_v3_pretrained2',
    'class4_pretrain_288_n25_2segmodels_v3',
    'class4_pretrain_288_n25_2segmodels_25epochs_v3',
    'class23_maxvit_224_2segmodels_v4_pretrained',
    'class4_pretrain_288_n25_2segmodels_v4',
    'class4_pretrain_288_n25_2segmodels_25epochs_v4',
    'class_all_pad30_maxvit_pretrained',
    'class_all_pad30_maxvit_crop2_pretrained',
    'class_all_pad30_convnext_pretrained',
    'class_all_pad30_maxvit_pretrained',
    'class_all_pad30_caformer',
    'class_all_pad30_seresnext_crop2_pretrained',
]
for config in configs:
    for fold in range(4):
        s = f'python train_one_fold.py -f {fold} -c {config}'
        print(s)
        os.system(s)

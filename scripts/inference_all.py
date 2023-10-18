import os
configs = [
    'class0_masked_192_2segmodels_pad0',
    'class0_masked_192_2segmodels_pad0_caformer',
    'class23_2segmodels_v3_xcit_small',
    'class0_gru_chaug_256_2segmodels_v3',
    'class1_lstm_112_2segmodels_20epochs_auc_v3_pretrained2',
    'class4_pretrain_288_n25_2segmodels_v3',
    'class4_pretrain_288_n25_2segmodels_25epochs_v3',
    'class23_maxvit_224_2segmodels_v4_pretrained',
    'class4_pretrain_288_n25_2segmodels_v4',
    'class4_pretrain_288_n25_2segmodels_25epochs_v4',
    'class_all_pad30_maxvit_pretrained_class1',
    'class_all_pad30_maxvit_crop2_pretrained_class0',
    'class_all_pad30_convnext_pretrained_class1',
    'class_all_pad30_maxvit_pretrained_class23',
    'class_all_pad30_caformer_class23',
    'class_all_pad30_seresnext_crop2_pretrained_class4',
]
for config in configs:
    for fold in range(4):
        s = f'python train_one_fold.py -f {fold} -c {config}'
        print(s)
        os.system(s)

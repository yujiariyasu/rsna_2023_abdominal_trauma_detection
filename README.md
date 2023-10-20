# RSNA 2023 Abdominal Trauma Detection 3rd place solution 

## prepare
```
pip install -r requirements.txt
```

## data download
- competition data
- png data from theo

```
cd input
. download.sh
```

## train & inference 3d segmentation
```
cd script
. train_3d_segmentation.sh
. inference_3d_segmentation.sh
```

## create cropped images
```
python create_cropped_images.py
python create_fold_df.py
```

## pretrain by image level label
```
python prepare_image_level_df.py
. pretrain.sh
```

## train & inference 2.5d+3d classification models
```
python train_all.py
python inference_all.py
```

## searce weights
```
python search_weights.py
```

## inference
see https://www.kaggle.com/code/yujiariyasu/3rd-place-inf-code#3d-segmentation

## weights & csvs
Below is the finished item
csvs:
https://www.kaggle.com/datasets/yujiariyasu/rsna2023-csvs
model weights:
https://www.kaggle.com/datasets/yujiariyasu/class23-maxvit-224-2segmodels-v4-pretrained
https://www.kaggle.com/datasets/yujiariyasu/class0-masked-192-2segmodels-pad0-caformer
https://www.kaggle.com/datasets/yujiariyasu/class-all-pad30-caformer
https://www.kaggle.com/datasets/yujiariyasu/class-all-pad30-maxvit-pretrained
https://www.kaggle.com/datasets/yujiariyasu/class-all-pad30-convnext-pretrained
https://www.kaggle.com/datasets/yujiariyasu/class-all-pad30-seresnext-crop2-pretrained
https://www.kaggle.com/datasets/yujiariyasu/class-all-pad30-maxvit-crop2-pretrained
https://www.kaggle.com/datasets/yujiariyasu/class1-lstm-112-20epochs-auc-v3-pretrained2-fix
https://www.kaggle.com/datasets/yujiariyasu/class23-2segmodels-v3-xcit-small
https://www.kaggle.com/datasets/yujiariyasu/class0-masked-192-2segmodels-pad0
https://www.kaggle.com/datasets/yujiariyasu/class4-pretrain-288-n25-2segmodels-25epochs-v4
https://www.kaggle.com/datasets/yujiariyasu/class4-pretrain-288-n25-2segmodels-v4
https://www.kaggle.com/datasets/yujiariyasu/class4-pretrain-288-n25-2segmodels-25epochs-v3
https://www.kaggle.com/datasets/yujiariyasu/class4-pretrain-288-n25-2segmodels-v3
https://www.kaggle.com/datasets/yujiariyasu/class0-gru-chaug-256-2segmodels-v3
https://www.kaggle.com/datasets/yujiariyasu/rsna-3dseg-resnet50-v2


my hardware:
CPU Intel Xeon Gold 6148 Processor（27.5 MB L3 Cache, 2.40 GHz, 20 Cores, 40 Threads）×2
GPU NVIDIA Tesla V100 SXM2 (16GiB HBM2)×4
Memory  384GiB DDR4 2666MHz RDIMM

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

hardware:
CPU Intel Xeon Gold 6148 Processor（27.5 MB L3 Cache, 2.40 GHz, 20 Cores, 40 Threads）×2
GPU NVIDIA Tesla V100 SXM2 (16GiB HBM2)×4
Memory  384GiB DDR4 2666MHz RDIMM

python train_one_fold.py -c rsna_pretrain_maxvit -f 0
python train_one_fold.py -c rsna_pretrain_maxvit -f 1
python train_one_fold.py -c rsna_pretrain_maxvit -f 2
python train_one_fold.py -c rsna_pretrain_maxvit -f 3

python train_one_fold.py -c rsna_pretrain_convnext -f 0
python train_one_fold.py -c rsna_pretrain_convnext -f 1
python train_one_fold.py -c rsna_pretrain_convnext -f 2
python train_one_fold.py -c rsna_pretrain_convnext -f 3

python train_one_fold.py -c rsna_pretrain_seresnext -f 0
python train_one_fold.py -c rsna_pretrain_seresnext -f 1
python train_one_fold.py -c rsna_pretrain_seresnext -f 2
python train_one_fold.py -c rsna_pretrain_seresnext -f 3

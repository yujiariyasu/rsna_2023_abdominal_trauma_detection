import os
import shutil
from multiprocessing import cpu_count
import sys
import datetime
import time
sys.path.append('../')
sys.path.append('/home/acc12347av/ml_pipeline')

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers.csv_logs import CSVLogger

import warnings
warnings.simplefilter('ignore')

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, default='Test', help="config name in configs.py")
    parser.add_argument("--type", '-t', type=str, default='classification')
    parser.add_argument("--debug", action='store_true', help="debug")
    parser.add_argument("--fold", '-f', type=int, default=0, help="fold num")
    return parser.parse_args()

if __name__ == "__main__":
    start = time.time()
    args = parse_args()
    if args.type == 'classification':
        from src.configs import *
    elif args.type == 'seg':
        from src.seg_configs import *

    try:
        cfg = eval(args.config)(args.fold)
    except:
        cfg = eval(args.config)()

    if cfg.inference_only:
        exit()
    if cfg.train_by_all_data & (args.fold != 0):
        exit()
    cfg.fold = args.fold
    if cfg.seed is None:
        now = datetime.datetime.now()
        cfg.seed = int(now.strftime('%s'))

    RESULTS_PATH_BASE = '../results'

    if args.type == 'classification':
        from src.lightning.lightning_modules.classification import MyLightningModule
        from src.lightning.data_modules.classification import MyDataModule
    elif args.type == 'seg':
        from src.lightning.lightning_modules.segmentation import MyLightningModule
        from src.lightning.data_modules.segmentation import MyDataModule

    if args.debug:
        cfg.epochs = 1
        cfg.n_cpu = 1
        n_gpu = 1
        dfs = []
    else:
        n_gpu = torch.cuda.device_count()
        cfg.n_cpu = n_gpu * np.min([cpu_count(), cfg.batch_size])

    print(f'\n----------------------- Config -----------------------')
    from datetime import datetime
    config_str = f'time: {datetime.now().strftime("%m-%d %H:%M")}, '
    for k, v in vars(cfg).items():
        if (k == 'model') | (k == 'teacher_models') | (('df' in k) & ('path' not in k)):
            continue
        if (k == 'label_features') and (len(v) > 100):
            print(f'\t{k} len: {len(v)}')
            config_str += f'{k} len: {len(v)}, '
        else:
            if type(v) != int: v = str(v).replace("\n", "")
            print(f'\t{k}: {v}')
            config_str += f'{k}: {v}, '
    print(f'----------------------- Config -----------------------\n')
    print('config_str:', config_str)

    if type(cfg.image_size) == int:
        cfg.image_size = (cfg.image_size, cfg.image_size)
    cfg.transform = cfg.transform(cfg.image_size)

    seed_everything(cfg.seed)
    OUTPUT_PATH = f'{RESULTS_PATH_BASE}/{args.config}'
    cfg.output_path = OUTPUT_PATH
    os.system(f'mkdir -p {cfg.output_path}/val_preds/fold{args.fold}')
    logger = CSVLogger(save_dir=OUTPUT_PATH, name=f"fold_{args.fold}")

    monitor = 'val_metric'
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_PATH, filename=f"fold_{args.fold}", auto_insert_metric_name=True,
        save_top_k=cfg.save_top_k, monitor=monitor, mode='max', verbose=True, save_last=True)
    checkpoint_callback.CHECKPOINT_NAME_LAST = f'last_fold{args.fold}'

    early_stop_callback = EarlyStopping(patience=cfg.early_stop_patience,
        monitor=monitor, mode='max', verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    if cfg.pretrained_path is not None:
        path = f'results/{cfg.pretrained_path}/fold_{args.fold}.ckpt'
        state_dict = torch.load(path, map_location=torch.device('cpu'))['state_dict']
        torch_state_dict = {}

        for k, v in state_dict.items():
            torch_state_dict[k[6:]] = v
        print(f'load model weight from checkpoint: {cfg.pretrained_path}')
        cfg.model.load_state_dict(torch_state_dict)

    if cfg.resume:
        find = False
        for i in range(100):
            i = 100 - i
            state_dict_path = f'{OUTPUT_PATH}/fold_{args.fold}-v{i}.ckpt'
            if os.path.exists(state_dict_path):
                find = True
                break
        if not find:
            state_dict_path = f'{OUTPUT_PATH}/fold_{args.fold}.ckpt'
    else:
        state_dict_path = None

    strategy = None
    trainer = Trainer(
        max_epochs=cfg.epochs,
        gpus=n_gpu,
        accumulate_grad_batches=cfg.grad_accumulations,
        precision=16 if cfg.fp16 else 32,
        amp_backend='native',
        deterministic=False,
        auto_select_gpus=False,
        benchmark=True,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=[logger],
        sync_batchnorm=cfg.sync_batchnorm,
        enable_progress_bar=True,
        resume_from_checkpoint=state_dict_path, # will be removed in v1.7. use ckpt_path instead of resume_from_checkpoint
        accelerator='gpu',
        strategy=strategy,
        devices=devices,
        reload_dataloaders_every_n_epochs=getattr(cfg, 'reload_dataloaders_every_n_epochs', 0),
    )
    model = MyLightningModule(cfg)
    datamodule = MyDataModule(cfg)
    print('start training.')
    trainer.fit(model, datamodule=datamodule)

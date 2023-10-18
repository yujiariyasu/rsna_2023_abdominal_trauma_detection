import torch
import random
import os
import numpy as np
import pandas as pd
import argparse
import warnings
warnings.simplefilter('ignore')
import sys
sys.path.append('../')
sys.path.append('/home/acc12347av/ml_pipeline')
from multiprocessing import cpu_count

def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, default='Test', help="config name in configs.py")
    parser.add_argument("--type", '-t', type=str, default='classification', help="type")
    parser.add_argument("--debug", action='store_true', help="debug")
    parser.add_argument("--fold", '-f', type=int, default=0, help="fold num")
    return parser.parse_args()

def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_args()

    if args.type == 'classification':
        from src.configs import *
    elif args.type == 'effdet':
        from src.effdet_configs import *

    try:
        cfg = eval(args.config)(args.fold)
    except:
        cfg = eval(args.config)()

    cfg.fold = args.fold
    cfg.batch_size *= 2

    if (not cfg.predict_valid) & (not cfg.predict_test):
        print('(not cfg.predict_valid) & (not cfg.predict_test)!')
        exit()

    RESULTS_PATH_BASE = '../results'

    if args.type == 'classification':
        if ('MetricLearning' in str(cfg.metric)) | cfg.arcface:
            from src.utils.predict_funcs import metric_learning_predict as predict
        else:
            from src.utils.predict_funcs import classification_predict as predict
        from src.utils.dataloader_factory import prepare_classification_loader as prepare_loader
    elif args.type == 'effdet':
        from src.utils.predict_funcs import effdet_predict as predict
        from src.utils.dataloader_factory import prepare_effdet_loader as prepare_loader
        from src.utils.effdet_model_factory import load_effdet_model
    elif args.type == 'seg':
        # from src.utils.predict_funcs import seg_predict_calc_metric as predict
        from src.utils.predict_funcs import seg_predict as predict
        from src.utils.dataloader_factory import prepare_seg_loader as prepare_loader
    elif args.type == 'nlp':
        from src.utils.predict_funcs import nlp_predict as predict
        from src.utils.dataloader_factory import prepare_nlp_loader as prepare_loader
    elif args.type == 'mlp':
        from src.utils.predict_funcs import mlp_predict as predict
        from src.utils.dataloader_factory import prepare_mlp_loader as prepare_loader
    elif args.type == 'mlp_with_nlp':
        if cfg.arcface:
            from src.utils.predict_funcs import metric_learning_mlp_with_nlp_predict as predict
        else:
            from src.utils.predict_funcs import mlp_with_nlp_predict as predict
        from src.utils.dataloader_factory import prepare_mlp_with_nlp_loader as prepare_loader
    elif args.type == 'segmentation':
        exit()

    if args.debug:
        cfg.n_cpu = 1
        n_gpu = 1
    else:
        n_gpu = torch.cuda.device_count()
        cfg.n_cpu = np.min([cpu_count(), cfg.batch_size])

    print(f'\n----------------------- Config -----------------------')
    config_str = ''
    for k, v in vars(cfg).items():
        if (k == 'model') | (('df' in k) & ('path' not in k)):
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

    if args.type not in ['nlp', 'mlp_with_nlp', 'mlp']:
        if type(cfg.image_size) == int:
            cfg.image_size = (cfg.image_size, cfg.image_size)
        cfg.transform = cfg.transform(cfg.image_size)
    seed_everything()
    if getattr(cfg, 'force_use_model_path_config_when_inf', False):
        load_model_config_dir = f'{RESULTS_PATH_BASE}/{cfg.force_use_model_path_config_when_inf}'
    else:
        load_model_config_dir = f'{RESULTS_PATH_BASE}/{args.config}'
    OUTPUT_PATH = f'{RESULTS_PATH_BASE}/{args.config}'
    os.system(f'mkdir -p {OUTPUT_PATH}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if getattr(cfg, 'use_last_ckpt_when_inference', False):
        file_name_base = 'last_fold'
    else:
        file_name_base = 'fold_'
    find = False
    for i in range(100):
        i = 100 - i
        state_dict_path = f'{load_model_config_dir}/{file_name_base}{args.fold}-v{i}.ckpt'
        if os.path.exists(state_dict_path):
            find = True
            break
    if not find:
        state_dict_path = f'{load_model_config_dir}/{file_name_base}{args.fold}.ckpt'

    if not getattr(cfg, 'no_trained_model_when_inf', False):
        state_dict = torch.load(state_dict_path)['state_dict']

        torch_state_dict = {}
        delete_model_model = True
        delete_model = True
        for k, v in state_dict.items():
            if not k.startswith('model.model.'):
                delete_model_model = False
            if not k.startswith('model.'):
                delete_model = False

        for k, v in state_dict.items():
            if delete_model_model:
                torch_state_dict[k[12:]] = v
            elif delete_model:
                torch_state_dict[k[6:]] = v
            else:
                torch_state_dict[k] = v


        print(f'load model weight from checkpoint: {state_dict_path}')
    if not getattr(cfg, 'no_trained_model_when_inf', False):
        if args.type == 'effdet':
            cfg.model = load_effdet_model(cfg, torch_state_dict)
        else:
            cfg.model.load_state_dict(torch_state_dict)
    cfg.model.to(device)

    if getattr(cfg, 'reset_classifier_when_inf', False):
        cfg.model.reset_classifier(0)

    if args.type == 'effdet':
        if cfg.predict_valid:
            val, val_loader = prepare_loader(cfg, predict_valid=True)
            paths = val.path.unique()
            preds = predict(cfg, val_loader)
            dfs = []
            for path, pred in zip(paths, preds):
                df = pd.DataFrame()
                df[['x_min','y_min','x_max','y_max','conf','class_id']] = pred
                df['path'] = path
                dfs.append(df)
            val = pd.concat(dfs)
            val.to_csv(f'{OUTPUT_PATH}/oof_fold{args.fold}.csv', index=False)

        if cfg.predict_test:
            test, test_loader = prepare_loader(cfg, predict_valid=False)
            paths = test.path.unique()
            preds = predict(cfg, test_loader)
            dfs = []
            for path, pred in zip(paths, preds):
                df = pd.DataFrame()
                df[['x_min','y_min','x_max','y_max','conf','class_id']] = pred
                df['path'] = path
                dfs.append(df)
            test = pd.concat(dfs)
            test.to_csv(f'{OUTPUT_PATH}/test_fold{args.fold}.csv', index=False)
    elif args.type == 'nlp':
        if cfg.predict_valid:
            val, val_loader = prepare_loader(cfg, predict_valid=True)
            if cfg.output_features:
                preds, features = predict(cfg, val_loader)
            else:
                preds = predict(cfg, val_loader)

            pred_cols = [f'pred_{c}' for c in cfg.label_features]
            # pred_cols = [f'pred_{i}' for i in range(8)]
            val[pred_cols] = preds
            val.to_csv(f'{OUTPUT_PATH}/oof_fold{args.fold}.csv', index=False)
            if cfg.output_features:
                np.save(f'{OUTPUT_PATH}/val_features_fold{args.fold}.npy', features)

        if cfg.predict_test:
            test, test_loader = prepare_loader(cfg, predict_valid=False)

            if cfg.output_features:
                preds, features = predict(cfg, test_loader)
            else:
                preds = predict(cfg, test_loader)

            pred_cols = [f'pred_{c}' for c in cfg.label_features]
            # pred_cols = [f'pred_{i}' for i in range(8)]
            test[pred_cols] = preds
            test.to_csv(f'{OUTPUT_PATH}/test_fold{args.fold}.csv', index=False)
            if cfg.output_features:
                np.save(f'{OUTPUT_PATH}/test_features_fold{args.fold}.npy', features)

    else:
        if cfg.predict_valid:
            val, val_loader = prepare_loader(cfg, split='val')
            if args.type == 'mlp_with_nlp':
                if cfg.arcface:
                    fs = predict(cfg, val_loader)
                    np.save(f'{OUTPUT_PATH}/val_features_fold{args.fold}.npy', fs)
                    print(f'val save to {OUTPUT_PATH}/val_features_fold{args.fold}.npy')
                else:
                    raise
            elif args.type == 'seg':
                os.system(f'mkdir {OUTPUT_PATH}/val_preds')
                # os.system(f'mkdir {OUTPUT_PATH}/val_targets')
                predict(cfg, val_loader, f'{OUTPUT_PATH}/val_preds')
            elif cfg.output_features:
                preds, features = predict(cfg, val_loader)

                # preds = np.mean(preds, axis=0)
                preds_n = 0
                for add_imsizes_n, add_imsizes in enumerate(cfg.add_imsizes_when_inference):
                    if add_imsizes_n == 0:
                        suffix = ''
                    else:
                        suffix = f'multi_scale_{add_imsizes_n}'
                    for tta_n in range(cfg.tta):
                        if tta_n != 0:
                            suffix += f'flip_{tta_n}'
                        if suffix != '':
                            pred_cols = [f'pred_{c}_{suffix}' for c in cfg.label_features]
                        val[pred_cols] = preds[preds_n]
                        preds_n += 1

                # np.save(f'{OUTPUT_PATH}/val_preds_fold{args.fold}.npy', preds)
                pred_cols = [f'pred_{c}' for c in cfg.label_features]
                val[pred_cols] = preds
                # val[cfg.train_df.individual_id.unique().tolist()] = preds
                val.to_csv(f'{OUTPUT_PATH}/oof_fold{args.fold}.csv', index=False)
                np.save(f'{OUTPUT_PATH}/val_features_fold{args.fold}.npy', features)
                print(f'val save to {OUTPUT_PATH}/oof_fold{args.fold}.csv')
                print(f'val features save to {OUTPUT_PATH}/val_features_fold{args.fold}.npy')
            else:
                preds = predict(cfg, val_loader)

                # preds = np.mean(preds, axis=0)
                # val = val.drop_duplicates('image_id')
                preds_n = 0
                for add_imsizes_n, add_imsizes in enumerate(cfg.add_imsizes_when_inference):
                    for tta_n in range(cfg.tta):
                        if add_imsizes_n == 0:
                            suffix = ''
                        else:
                            suffix = f'multi_scale_{add_imsizes_n}_'
                        if tta_n != 0:
                            suffix += f'flip_{tta_n}'
                        if suffix == '':
                            pred_cols = [f'pred_{c}' for c in cfg.label_features]
                        else:
                            pred_cols = [f'pred_{c}_{suffix}' for c in cfg.label_features]
                        val[pred_cols] = preds[preds_n]
                        preds_n += 1
                # val[cfg.train_df.individual_id.unique().tolist()] = preds
                val.to_csv(f'{OUTPUT_PATH}/oof_fold{args.fold}.csv', index=False)
                print(f'val save to {OUTPUT_PATH}/oof_fold{args.fold}.csv')

        if cfg.predict_test:
            test, test_loader = prepare_loader(cfg, split='test')
            if args.type == 'mlp_with_nlp':
                if cfg.arcface:
                    fs = predict(cfg, test_loader)
                    np.save(f'{OUTPUT_PATH}/test_features_fold{args.fold}.npy', fs)
                    print(f'test save to {OUTPUT_PATH}/test_features_fold{args.fold}.npy')
                else:
                    raise
            elif args.type == 'seg':
                os.system(f'mkdir {OUTPUT_PATH}/test_preds_fold{args.fold}')
                predict(cfg, test_loader, f'{OUTPUT_PATH}/test_preds_fold{args.fold}')
            elif cfg.output_features:
                preds, features = predict(cfg, test_loader)
                preds = np.mean(preds, axis=0)
                # np.save(f'{OUTPUT_PATH}/test_preds_fold{args.fold}.npy', preds)
                pred_cols = [f'pred_{c}' for c in cfg.label_features]
                test[pred_cols] = preds
                test.to_csv(f'{OUTPUT_PATH}/test_fold{args.fold}.csv', index=False)
                np.save(f'{OUTPUT_PATH}/test_features_fold{args.fold}.npy', features)
                print(f'test save to {OUTPUT_PATH}/test_fold{args.fold}.csv')
                print(f'test save to {OUTPUT_PATH}/test_features_fold{args.fold}.npy')
            else:
                preds = predict(cfg, test_loader)
                # preds = np.mean(preds, axis=0)
                preds_n = 0
                for add_imsizes_n, add_imsizes in enumerate(cfg.add_imsizes_when_inference):
                    for tta_n in range(cfg.tta):
                        if add_imsizes_n == 0:
                            suffix = ''
                        else:
                            suffix = f'multi_scale_{add_imsizes_n}_'
                        if tta_n != 0:
                            suffix += f'flip_{tta_n}'
                        if suffix == '':
                            pred_cols = [f'pred_{c}' for c in cfg.label_features]
                        else:
                            pred_cols = [f'pred_{c}_{suffix}' for c in cfg.label_features]
                        test[pred_cols] = preds[preds_n]
                        preds_n += 1

                # np.save(f'{OUTPUT_PATH}/test_preds_fold{args.fold}.npy', preds)
                # pred_cols = [f'pred_{c}' for c in cfg.label_features]
                # test[pred_cols] = preds
                test.to_csv(f'{OUTPUT_PATH}/test_fold{args.fold}.csv', index=False)
                print(f'test save to {OUTPUT_PATH}/test_fold{args.fold}.csv')

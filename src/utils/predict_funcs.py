from scipy.special import softmax
import torch
from tqdm import tqdm
import numpy as np
from pdb import set_trace as st
import torchvision.transforms.functional as F

# def classification_predict(cfg, loader):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     cfg.model.eval()
#     with torch.no_grad():
#         tta_predictions = []
#         for add_imsize in cfg.add_imsizes_when_inference:
#             for tta_n in range(cfg.tta):
#                 predictions = []
#                 features_list = []
#                 for images in tqdm(loader):
#                     if (add_imsize[0] != 0) | (add_imsize[1] != 0):
#                         images = F.resize(img=images, size=(cfg.image_size[0]+add_imsize[0], cfg.image_size[1]+add_imsize[1]))
#                     if len(cfg.meta_cols) != 0:
#                         images, meta = images
#                         images, meta = images.to(device), meta.to(device)
#                         images = (images, meta)
#                     else:
#                         images = images.to(device)
#                     if tta_n % 2 == 1:
#                         images = torch.flip(images, (3,))
#                     if tta_n % 4 >= 2:
#                         images = torch.flip(images, (2,))
#                     if tta_n % 8 >= 4:
#                         images = torch.transpose(images, 2,3)
#                     if cfg.output_features:
#                         pred, features = cfg.model.extract_with_features(images)
#                     else:
#                         pred = cfg.model(images)
#                     if cfg.output_features:
#                         features_list.append(features.detach())
#                     predictions.append(pred.detach())
#                 tta_predictions.append(torch.cat(predictions).cpu().numpy())
#     if cfg.output_features:
#         return tta_predictions, torch.cat(features_list).cpu().numpy()
#     else:
#         return tta_predictions


def classification_predict(cfg, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg.inf_fp16:
        cfg.model.half().eval()
    else:
        cfg.model.eval()
    tta_predictions = [[] for _ in range(len(cfg.add_imsizes_when_inference)*cfg.tta)]
    assert cfg.tta <= 8
    with torch.no_grad():
        for images_n, images in enumerate(tqdm(loader)):
            if cfg.inf_fp16:
                images = images.half()
            tta_n = 0
            if getattr(cfg, 'multi_image_4classes', False):
                images = [i.to(device) for i in images]
            elif len(cfg.meta_cols) != 0:
                images, meta = images
                images, meta = images.to(device), meta.to(device)
            else:
                try:
                    images = images.to(device)
                except:
                    images = [i.to(device) for i in images]

            for add_imsize in cfg.add_imsizes_when_inference:
                predictions = []
                features_list = []
                if (add_imsize[0] != 0) | (add_imsize[1] != 0):
                    images = F.resize(img=images, size=(cfg.image_size[0]+add_imsize[0], cfg.image_size[1]+add_imsize[1]))
                if images_n == 0:
                    try:
                        print('images.size():', images.size())
                    except:
                        pass
                for flip_tta_n in range(cfg.tta):
                    if flip_tta_n % 2 == 1:
                        images = torch.flip(images, (3,))
                    if flip_tta_n in [2, 6]:
                        images = torch.flip(images, (2,))
                    if flip_tta_n == 4:
                        images = torch.transpose(images, 2,3)

                    input_ = images if len(cfg.meta_cols) == 0 else (images, meta)
                    if cfg.output_features:
                        pred, features = cfg.model.extract_with_features(input_)
                    else:
                        pred = cfg.model(input_)
                    if cfg.output_features:
                        features_list.append(features.detach())
                    tta_predictions[tta_n] += pred.detach().cpu().numpy().tolist()
                    # tta_predictions[tta_n].append(torch.cat(predictions))
                    tta_n += 1
    if cfg.output_features:
        return tta_predictions, torch.cat(features_list).cpu().numpy()
    else:
        return tta_predictions

def mlp_with_nlp_predict(cfg, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.model.eval()
    with torch.no_grad():
        tta_predictions = []
        for tta_n in range(cfg.tta):
            predictions = []
            for images in tqdm(loader):
                images = images.to(device)
                if tta_n % 2 == 1:
                    images = torch.flip(images, (3,))
                if tta_n % 4 >= 2:
                    images = torch.flip(images, (2,))
                if tta_n % 8 >= 4:
                    images = torch.transpose(images, 2,3)
                pred = cfg.model(images)
                # features_list.append(features.detach())
                predictions.append(pred.detach())
            tta_predictions.append(torch.cat(predictions).cpu().numpy())
    return tta_predictions

def mlp_predict(cfg, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.model.eval()
    with torch.no_grad():
        tta_predictions = []
        for tta_n in range(cfg.tta):
            predictions = []
            for metas in tqdm(loader):
                metas = metas.to(device)
                pred = cfg.model(metas)
                predictions.append(pred.detach())
            tta_predictions.append(torch.cat(predictions).cpu().numpy())
    return tta_predictions

def seg_predict(cfg, loader, save_dir=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.model.eval()
    with torch.no_grad():
        tta_predictions = []
        for tta_n in range(cfg.tta):
            predictions = []
            for images, ids in tqdm(loader):
                images = images.to(device)
                if tta_n % 2 == 1:
                    images = torch.flip(images, (3,))
                if tta_n % 4 >= 2:
                    images = torch.flip(images, (2,))
                if tta_n % 8 >= 4:
                    images = torch.transpose(images, 2,3)
                pred = cfg.model(images)
                if tta_n % 2 == 1:
                    pred = torch.flip(pred, (3,))
                if tta_n % 4 >= 2:
                    pred = torch.flip(pred, (2,))
                if tta_n % 8 >= 4:
                    pred = torch.transpose(pred, 2,3)
                # features_list.append(features.detach())
                # predictions.append(pred.detach().cpu())
                if cfg.save_preds:
                    for pr, id in zip(pred.detach().cpu().numpy(), ids):
                        np.save(f'{save_dir}/{id}.npy', pr)
                # if cfg.save_targets:
                #     for im, id in zip(masks.detach().cpu().numpy(), ids):
                #         np.save(f'{save_dir.replace('preds', 'targets')}/{id}.npy', im)

            # tta_predictions.append(torch.cat(predictions).numpy())
    # return np.mean(tta_predictions, axis=0)

def seg_predict_calc_metric(cfg, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.model.eval()
    with torch.no_grad():
        tta_predictions = []
        targets = []
        for tta_n in range(cfg.tta):
            predictions = []
            for images, masks in tqdm(loader):
                images = images.to(device)
                if tta_n % 2 == 1:
                    images = torch.flip(images, (3,))
                if tta_n % 4 >= 2:
                    images = torch.flip(images, (2,))
                if tta_n % 8 >= 4:
                    images = torch.transpose(images, 2,3)
                pred = cfg.model(images)
                if tta_n % 2 == 1:
                    pred = torch.flip(pred, (3,))
                if tta_n % 4 >= 2:
                    pred = torch.flip(pred, (2,))
                if tta_n % 8 >= 4:
                    pred = torch.transpose(pred, 2,3)
                # features_list.append(features.detach())
                predictions.append(pred.detach().cpu())
                if tta_n == 0:
                    targets.append(masks)
            tta_predictions.append(torch.cat(predictions))
    targets = torch.cat(targets)
    preds = torch.mean(tta_predictions, axis=0)
    # score = cfg.metric(targets, preds)
    # print('score:', score)

    return preds.numpy(), targets.numpy()

def metric_learning_mlp_with_nlp_predict(cfg, loader, fliplr=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.model.eval()
    with torch.no_grad():
        features = []
        for input in tqdm(loader):
            ids = input['ids'].to(device, dtype=torch.long)
            mask = input['mask'].to(device, dtype=torch.long)
            num_vals = input['num_vals'].to(device, dtype=torch.float)
            feature = cfg.model.extract(ids, mask, num_vals)
            features.append(feature.detach())
    return torch.cat(features).cpu().numpy()

def nlp_predict(cfg, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.model.eval()
    with torch.no_grad():
        predictions = []
        features_list = []
        for ids, masks, token_type_ids in tqdm(loader):
            ids, masks, token_type_ids = ids.to(device), masks.to(device), token_type_ids.to(device)
            if cfg.output_features:
                pred, features = cfg.model.forward_with_features(ids, masks, token_type_ids)
            else:
                pred = cfg.model(ids, masks, token_type_ids)
            predictions.append(pred.detach())
            if cfg.output_features:
                features_list.append(features.detach())
    if cfg.output_features:
        return torch.cat(predictions).cpu().numpy(), torch.cat(features_list).cpu().numpy()
    return torch.cat(predictions).cpu().numpy()

def effdet_predict(cfg, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.model.eval()
    with torch.no_grad():
        tta_predictions = []
        for tta_n in range(cfg.tta):
            predictions = []
            for images in tqdm(loader):
                images = images.to(device)
                # if tta_n % 2 == 1:
                #     images = torch.flip(images, (3,))
                # if tta_n % 4 >= 2:
                #     images = torch.flip(images, (2,))
                # if tta_n % 8 >= 4:
                #     images = torch.transpose(images, 2,3)
                pred = cfg.model(images)
                # features_list.append(features.detach())
                predictions.append(pred.detach())
            return torch.cat(predictions).cpu().numpy()
            tta_predictions.append(torch.cat(predictions).cpu().numpy())
    return tta_predictions

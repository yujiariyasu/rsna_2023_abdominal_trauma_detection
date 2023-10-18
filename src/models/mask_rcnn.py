import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_maskrcnn_resnet50_fpn(num_classes, pretrained=True, model_ckpt=None, pretrained_num_classes=1):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained,
                                                               box_detections_per_img=1000)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, pretrained_num_classes+1)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, pretrained_num_classes+1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_ckpt:
        model.load_state_dict(torch.load(model_ckpt, map_location=device)['model'])

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes+1)

        # model.module.load_state_dict(checkpoint['model'])

    return model

from effdet import get_efficientdet_config, create_model_from_config#, EfficientDet, DetBenchTrain

def load_effdet_model(cfg, torch_state_dict=None, bench_task='predict', bench_labeler=False):
    num_classes = cfg.num_classes
    image_size = cfg.image_size
    model_name = cfg.model_name
    base_config = get_efficientdet_config(model_name)
    if type(image_size) == int:
        image_size = (image_size, image_size)
    base_config.image_size = image_size

    if(torch_state_dict):
        model = create_model_from_config(
            base_config,
            bench_task=bench_task,
            num_classes=num_classes,
            pretrained=True,
            pretrained_backbone=False,
            bench_labeler=bench_labeler
        )
        # import pdb;pdb.set_trace()
        try:
            model.load_state_dict(torch_state_dict)
        except:
            del torch_state_dict['anchors.boxes']
            model.load_state_dict(torch_state_dict, strict=False)
    else:
        model = create_model_from_config(
            base_config,
            bench_task=bench_task,
            pretrained=True,
            num_classes=num_classes,
            bench_labeler=bench_labeler
        )
    return model

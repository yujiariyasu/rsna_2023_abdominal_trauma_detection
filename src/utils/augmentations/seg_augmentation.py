def seg_aug_base_v1(size):
    return {
        'train': A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # A.ShiftScaleRotate(),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "val": A.Compose([
            # A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            # A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ], p=1.0)}

def seg_aug_base_v2(size):
    return {
        'train': A.Compose([
            # A.Resize(size, size)
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Transpose(),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "val": A.Compose([
            # A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            # A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ], p=1.0)}


def seg_aug_base_v3(size):
    return {
        'train': A.Compose([
                    A.RandomRotate90(),
                    A.Flip(),
                    A.Transpose(),
                    A.GaussNoise(),
                    ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "val": A.Compose([
            # A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            # A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ], p=1.0)}

def seg_aug_base_v4(size):
    return {
        'train': A.Compose([
                    A.RandomRotate90(),
                    A.Flip(),
                    A.Transpose(),
                    # A.GaussNoise(),
                    A.OneOf([
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ], p=0.5),
                    # A.OneOf([
                    #     A.IAASharpen(),
                    #     A.IAAEmboss(),
                    #     A.RandomBrightnessContrast(),
                    # ], p=0.5),
                    # A.HueSaturationValue(p=0.3),
                    ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "val": A.Compose([
            # A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            # A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ], p=1.0)}

def seg_aug_base_v5(size):
    return {
        'train': A.Compose([
                    A.RandomRotate90(),
                    A.Flip(),
                    A.Transpose(),
                    # A.GaussNoise(),
                    # A.OneOf([
                    #     A.MotionBlur(p=0.2),
                    #     A.MedianBlur(blur_limit=3, p=0.1),
                    #     A.Blur(blur_limit=3, p=0.1),
                    # ], p=0.5),
                    A.OneOf([
                        A.IAASharpen(),
                        A.IAAEmboss(),
                        A.RandomBrightnessContrast(),
                    ], p=0.5),
                    # A.HueSaturationValue(p=0.3),
                    ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "val": A.Compose([
            # A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            # A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ], p=1.0)}

def seg_aug_base_v6(size):
    return {
        'train': A.Compose([
                    A.RandomRotate90(),
                    A.Flip(),
                    A.Transpose(),
                    # A.GaussNoise(),
                    # A.OneOf([
                    #     A.MotionBlur(p=0.2),
                    #     A.MedianBlur(blur_limit=3, p=0.1),
                    #     A.Blur(blur_limit=3, p=0.1),
                    # ], p=0.5),
                    # A.OneOf([
                    #     A.IAASharpen(),
                    #     A.IAAEmboss(),
                    #     A.RandomBrightnessContrast(),
                    # ], p=0.5),
                    A.HueSaturationValue(p=0.3),
                    ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "val": A.Compose([
            # A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            # A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ], p=1.0)}


def seg_aug_base_v2_without_resize(size):
    return {
        'train': A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "val": A.Compose([
            # A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            # A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ], p=1.0)}

def seg_no_aug(size=None):
    return {
        'train': A.Compose([
            # A.HorizontalFlip(),
            # A.VerticalFlip(),
            # A.ShiftScaleRotate(),
            # A.Resize(size,size,always_apply=True),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "val": A.Compose([
            # A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            # A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(),
        ], p=1.0)}

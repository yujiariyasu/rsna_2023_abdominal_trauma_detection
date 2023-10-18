import numpy as np
import torch

class BaseEffdetTTA:
    def augment(self, images):
        raise NotImplementedError

    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)
        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)
        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)
        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)
        return result_boxes

    def deaugment_boxes(self, boxes):
        raise NotImplementedError

class TTAHorizontalFlip(BaseEffdetTTA):
    def __init__(self, image_size):
        self.image_size = image_size

    def fasterrcnn_augment(self, images):
        return list(image.flip(1) for image in images)

    def effdet_augment(self, images):
        return images.flip(2)

    def deaugment_boxes(self, boxes):
        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]
        return self.prepare_boxes(boxes)

class TTAVerticalFlip(BaseEffdetTTA):
    def __init__(self, image_size):
        self.image_size = image_size

    def fasterrcnn_augment(self, images):
        return list(image.flip(2) for image in images)

    def effdet_augment(self, images):
        return images.flip(3)

    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]
        return boxes

class TTARotate90(BaseEffdetTTA):
    def __init__(self, image_size):
        self.image_size = image_size

    def fasterrcnn_augment(self, images):
        return list(torch.rot90(image, 1, (1, 2)) for image in images)

    def effdet_augment(self, images):
        return torch.rot90(images, 1, (2, 3))

    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = self.image_size - boxes[:, [1,3]]
        res_boxes[:, [1,3]] = boxes[:, [2,0]]
        return self.prepare_boxes(res_boxes)

class TTACompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def fasterrcnn_augment(self, images):
        for transform in self.transforms:
            images = transform.fasterrcnn_augment(images)
        return images

    def effdet_augment(self, images):
        for transform in self.transforms:
            images = transform.effdet_augment(images)
        return images

    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        import pdb;pdb.set_trace()
        result_boxes[:,0] = torch.min(boxes[:, [0,2]], axis=1)
        result_boxes[:,2] = torch.max(boxes[:, [0,2]], axis=1)
        result_boxes[:,1] = torch.min(boxes[:, [1,3]], axis=1)
        result_boxes[:,3] = torch.max(boxes[:, [1,3]], axis=1)
        return result_boxes

    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)

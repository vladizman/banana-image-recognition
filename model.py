import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn as FasterRCNN
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights as Weights
from torchvision.models.detection.faster_rcnn import _fasterrcnn_mobilenet_v3_large_fpn as MobileNetV3_Large
from torchvision.models.detection.faster_rcnn import FasterRCNN_MobileNet_V3_Large_FPN_Weights as MobileNetWeights

def build_model(backbone: str):
    if backbone == 'fasterrcnn_resnet50_fpn':
        weights = Weights.DEFAULT
        model = FasterRCNN(pretrained=True, weights=weights)

    else:
        weights = MobileNetWeights.DEFAULT
        model = MobileNetV3_Large(pretrained=True, weights=weights)

    return model



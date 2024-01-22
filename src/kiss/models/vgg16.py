import torch.nn as nn
from torchvision import models


def vgg16_kiss(num_classes: int = 100):
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
    model.classifier = nn.Sequential()
    model.classifier.add_module('linear1', nn.Linear(in_features=2048, out_features=1024))
    model.classifier.add_module('relu1', nn.ReLU(inplace=True))
    model.classifier.add_module('dropout1', nn.Dropout(p=0.5))
    model.classifier.add_module('linear2', nn.Linear(in_features=1024, out_features=512))
    model.classifier.add_module('relu2', nn.ReLU(inplace=True))
    model.classifier.add_module('dropout2', nn.Dropout(p=0.5))
    model.classifier.add_module('linear3', nn.Linear(in_features=512, out_features=num_classes))
    
    return model
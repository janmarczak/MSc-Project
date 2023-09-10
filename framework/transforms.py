
import torch
import torch.nn as nn
import torchvision.transforms as transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms_for_eval() -> nn.Module:

    transform_list = [
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((224,224), antialias=False),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]

    return nn.Sequential(*transform_list)

def get_transforms_for_train(augment: bool = True) -> nn.Module:
    if not augment:
        return get_transforms_for_eval()

    transform_list = [
        transforms.ConvertImageDtype(torch.float32),
        transforms.RandomResizedCrop(224, antialias=False),
        transforms.RandomRotation(15),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]

    return nn.Sequential(*transform_list)
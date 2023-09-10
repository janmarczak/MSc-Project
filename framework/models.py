
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models


class Resnet(nn.Module):
    """
    Resnet model with a prediction head and optional 
    adapters for knowledge distillation used throughout this study
    """
    def __init__(self, model_name, num_classes, pretrained=True, student=False, teacher_hint_channels=None, feature_maps_at=[False, False, False, False]):
        super(Resnet, self).__init__()
        if model_name == 'resnet18':
            if pretrained: resnet = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            else: resnet = models.resnet18(weights=None)
        elif model_name == 'resnet34':
            if pretrained: resnet = models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
            else: resnet = models.resnet34(weights=None)
        elif model_name == 'resnet50':
            if pretrained: resnet = models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            else: resnet = models.resnet50(weights=None)
        elif model_name == 'resnet101':
            if pretrained: resnet = models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
            else: resnet = models.resnet101(weights=None)
        elif model_name == 'resnet152':
            if pretrained: resnet = models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
            else: resnet = models.resnet152(weights=None)
        else:
            raise ValueError('Invalid Resnet model name')
        
        self.student = student
        self._backbone = resnet
        features_num = self._backbone.fc.in_features
        self._backbone.fc = nn.Identity()
        self._prediction_head = nn.Linear(features_num, num_classes)

        self.feature_maps = []
        self.ft_layers = [name for name, selected in zip(['layer1', 'layer2', 'layer3', 'layer4'], feature_maps_at) if selected]
        self.register_hooks()

        self.adapters = nn.ModuleList()
        if self.student:
            # Need as many adapters as feature maps
            output_dims = self._infer_feature_dim()
            for i, layer in enumerate(self.ft_layers):
                self.adapters.append(
                    nn.Conv2d(output_dims[i], teacher_hint_channels[i], kernel_size=1, stride=1, padding=0, bias=False)
                )

    def register_hooks(self):
        def hook_fn(m, i, o):
            self.feature_maps.append(o)

        for name, module in self._backbone.named_modules():
            if name in self.ft_layers:
                module.register_forward_hook(hook_fn)

    
    def compute_attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), 2)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2,3), keepdim=True) # L2 norm across spatial dimensions
        am = torch.div(am, norm+eps)
        return am


    def forward(self, x):
        self.feature_maps = []
        backbone_output = self._backbone(x)
        logits = self._prediction_head(backbone_output)
        
        attention_maps = [self.compute_attention_map(fmap) for fmap in self.feature_maps]
        
        if self.student:
            adapted_features = [adapter(fmap) for adapter, fmap in zip(self.adapters, self.feature_maps)]
            return adapted_features, attention_maps, logits
        else:
            return self.feature_maps, attention_maps, logits
        

    def _infer_feature_dim(self):
        self.hook_output_dims = []
        # Define a forward hook
        def hook_fn(m, i, o):
            # Save the output dimensions
            self.hook_output_dims.append(o.size(1))

        hooks = []
        for layer_name in self.ft_layers:
            for name, module in self._backbone.named_modules():
                if name == layer_name:
                    hooks.append(module.register_forward_hook(hook_fn))

        # Forward an example tensor through the model
        x = torch.randn(1, 3, 224, 224)
        _ = self._backbone(x)

        # Unregister the hooks
        for hook in hooks:
            hook.remove()
        
        # Return the output dimensions
        return self.hook_output_dims
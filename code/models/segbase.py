"""Base Model for Semantic Segmentation"""
import torch
import torch.nn as nn
import math
from .base_models.resnet import *

__all__ = ['SegBaseModel']

def get_gaussian_filter(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    if kernel_size == 3:
        padding = 1
    else:
        padding = 0
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter

class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, aux, backbone='resnet50', task_len=None,local_rank=None, pretrained_base=True, epoch_count=None, img_size=None, **kwargs):
        super(SegBaseModel, self).__init__()
        self.aux = aux
        self.nclass = nclass
        self.backbone = backbone
        self.epoch, self.std = 5, 1
        self.epoch_count = epoch_count
        if backbone == 'resnet18':
            self.pretrained = resnet18_v1s(pretrained=pretrained_base, dilated=True, local_rank=local_rank, **kwargs)
        elif backbone == 'resnet50':
            self.pretrained = resnet50_v1s(pretrained=pretrained_base, local_rank=local_rank, dilated=True, **kwargs)
        elif backbone == 'resnet101':
            self.pretrained = resnet101_v1s(pretrained=pretrained_base, local_rank=local_rank, dilated=True, **kwargs)
        
        elif backbone == 'resnet18_original':
            self.pretrained = resnet50_v1b(pretrained=pretrained_base, dilated=True, local_rank=local_rank, **kwargs)
        elif backbone == 'resnet50_original':
            self.pretrained = resnet50_v1b(pretrained=pretrained_base, local_rank=local_rank, dilated=True, **kwargs)
        elif backbone == 'resnet101_original':
            self.pretrained = resnet101_v1b(pretrained=pretrained_base, local_rank=local_rank, dilated=True, **kwargs)

        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def get_new_kernels(self):
        if self.epoch_count % self.epoch == 0 and self.epoch_count != 0:
            self.std *= 0.9

        self.kernel1 = get_gaussian_filter(
                kernel_size=self.kernel_size,
                sigma=self.std,
                channels=64
        )
        self.kernel2= get_gaussian_filter(
                kernel_size=self.kernel_size,
                sigma=self.std,
                channels=128
        )
        self.kernel3 = get_gaussian_filter(
                kernel_size=self.kernel_size,
                sigma=self.std,
                channels=256
        )
        self.kernel4 = get_gaussian_filter(
                kernel_size=self.kernel_size,
                sigma=self.std,
                channels=512
        )
        self.kernel5 = get_gaussian_filter(
                kernel_size=self.kernel_size,
                sigma=self.std,
                channels=512
        )

    def base_forward(self, x):
        """forwarding pre-trained network"""
        
        if self.backbone.split('_')[-1] == 'original':
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
        else:
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu1(x)

            x = self.pretrained.conv2(x)
            x = self.pretrained.bn2(x)
            x = self.pretrained.relu2(x)

            x = self.pretrained.conv3(x)
            x = self.pretrained.bn3(x)
            x = self.pretrained.relu3(x)
            x = self.pretrained.maxpool(x)
        
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred


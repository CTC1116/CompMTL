"""Model store which handles pretrained models """

from .pspnet import *
from .deeplabv3 import *
from .deeplabv3_mobile import *
from .psp_mobile import *
from .segformer import *
from .lenet import *
from .clip_model import *

__all__ = ['get_segmentation_model']


def get_segmentation_model(model, **kwargs):
    models = {
        'psp': get_psp,
        'deeplabv3': get_deeplabv3,
        'deeplabv3_multi': get_deeplabv3_multi,
        'deeplabv3_multi_kd': get_deeplabv3_multi,
        'deeplabv3_mtan': get_deeplabv3_mtan,
        'deeplab_mobile': get_deeplabv3_mobile,
        'deeplab_mobile_multi': get_deeplabv3_mobile_multi,
        'psp_mobile': get_psp_mobile,
        'segformer': get_segformer,
        'segformer_multi':get_segformer_multi,
        'lenet': get_lenet,
        'lenet_multi': get_lenet_multi,
        'clip_text': get_clip_text,
    }
    return models[model](**kwargs)

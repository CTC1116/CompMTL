"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .base_models.resnet import conv1x1, BottleneckV1b

__all__ = ['get_deeplabv3', 'get_deeplabv3_multi', 'get_deeplabv3_mtan']

class DeepLabV3_mtan(SegBaseModel):
    r"""DeepLabV3_multi

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).
    """

    def __init__(self, nclass, backbone='resnet50', aux=False, local_rank=None, pretrained_base=True, **kwargs):
        super(DeepLabV3_mtan, self).__init__(nclass,aux, backbone,  local_rank, pretrained_base=pretrained_base, **kwargs)
        if backbone == 'resnet18':
            in_channels = 512
        else:
            in_channels = 2048
        ch = [256, 512, 1024, 2048]
        backbone = self.pretrained
        
        self.tasks = ['segmentation', 'depth', 'normal']
        self.num_out_channels = {'segmentation': nclass, 'depth': 1, 'normal': 3}
        
        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.conv2, backbone.bn2, backbone.relu2, backbone.conv3, backbone.bn3, backbone.relu3, backbone.maxpool)
        
        # We will apply the attention over the last bottleneck layer in the ResNet. 
        self.shared_layer1_b = backbone.layer1[:-1] 
        self.shared_layer1_t = backbone.layer1[-1]

        self.shared_layer2_b = backbone.layer2[:-1]
        self.shared_layer2_t = backbone.layer2[-1]

        self.shared_layer3_b = backbone.layer3[:-1]
        self.shared_layer3_t = backbone.layer3[-1]

        self.shared_layer4_b = backbone.layer4[:-1]
        self.shared_layer4_t = backbone.layer4[-1]

        # Define task specific attention modules using a similar bottleneck design in residual block
        # (to avoid large computations)
        self.encoder_att_1 = nn.ModuleList([self.att_layer(ch[0], ch[0] // 4, ch[0]) for _ in self.tasks])
        self.encoder_att_2 = nn.ModuleList([self.att_layer(2 * ch[1], ch[1] // 4, ch[1]) for _ in self.tasks])
        self.encoder_att_3 = nn.ModuleList([self.att_layer(2 * ch[2], ch[2] // 4, ch[2]) for _ in self.tasks])
        self.encoder_att_4 = nn.ModuleList([self.att_layer(2 * ch[3], ch[3] // 4, ch[3]) for _ in self.tasks])

        # Define task shared attention encoders using residual bottleneck layers
        # We do not apply shared attention encoders at the last layer,
        # so the attended features will be directly fed into the task-specific decoders.
        self.encoder_block_att_1 = self.conv_layer(ch[0], ch[1] // 4)
        self.encoder_block_att_2 = self.conv_layer(ch[1], ch[2] // 4)
        self.encoder_block_att_3 = self.conv_layer(ch[2], ch[3] // 4)
        
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList([_DeepLabHead(in_channels, self.num_out_channels[t]) for t in self.tasks])
        if self.aux:
            self.auxlayers = nn.ModuleList([_FCNHead(in_channels, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x):
        # Shared convolution
        x = self.shared_conv(x)
        
        # Shared ResNet block 1
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)

        # Shared ResNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)

        # Shared ResNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)
        
        # Shared ResNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)

        # Attention block 1 -> Apply attention over last residual block
        a_1_mask = [att_i(u_1_b) for att_i in self.encoder_att_1]  # Generate task specific attention map
        a_1 = [a_1_mask_i * u_1_t for a_1_mask_i in a_1_mask]  # Apply task specific attention map to shared features
        a_1 = [self.down_sampling(self.encoder_block_att_1(a_1_i)) for a_1_i in a_1]
        
        # Attention block 2 -> Apply attention over last residual block
        a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)]
        a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [self.encoder_block_att_2(a_2_i) for a_2_i in a_2]
        
        # Attention block 3 -> Apply attention over last residual block
        a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
        a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [self.encoder_block_att_3(a_3_i) for a_3_i in a_3]
        
        # Attention block 4 -> Apply attention over last residual block (without final encoder)
        a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_att_4)]
        a_4 = [a_4_mask_i * u_4_t for a_4_mask_i in a_4_mask]
        
        # Task specific decoders
        out = [0 for _ in self.tasks]
        aux = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i], x_feat_after_aspp = self.decoders[i](a_4[i])
            aux[i] = self.auxlayers[i](a_3[i])
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
                aux[i] = aux[i] / torch.norm(aux[i], p=2, dim=1, keepdim=True)
        
        original_list = [out, aux]
        original_list = [list(i) for i in zip(*original_list)]
        original_list.append(a_4)
        return original_list
    
    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid())
        
    def conv_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(conv1x1(in_channel, 4 * out_channel, stride=1),
                                   nn.BatchNorm2d(4 * out_channel))
        return BottleneckV1b(in_channel, out_channel, downsample=downsample)
class Resnet_multi(SegBaseModel): 
    r"""DeepLabV3_multi

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).
    """

    def __init__(self, nclass, backbone='resnet50', aux=False, task_len=None,local_rank=None, pretrained_base=True, **kwargs):
        super(DeepLabV3_multi, self).__init__(nclass,aux, backbone,  task_len, local_rank, pretrained_base=pretrained_base, **kwargs)
        self.aux = aux
        self.task = 'multi'
        self.task_len = task_len
        
        if backbone == 'resnet18':
            in_channels = 512
        else:
            in_channels = 2048

        self.head = _DeepLabHead(in_channels, nclass, **kwargs)
        if self.aux:
            self.auxlayer = _FCNHead(in_channels // 2, nclass, **kwargs)
        if self.task=='multi':
            self.head1 = _DeepLabHead(in_channels, 1, **kwargs)
            self.head2 = _DeepLabHead(in_channels, 3, **kwargs)
            if self.aux:
                self.auxlayer1 = _FCNHead(in_channels // 2, 1, **kwargs)
                self.auxlayer2 = _FCNHead(in_channels // 2, 3, **kwargs)
        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])
    def forward(self, x):
        size = x.size()[2:]
        # auxout, auxout1, auxout2 = torch.zeros(size), torch.zeros(size), torch.zeros(size)
        c1, c2, c3, c4 = self.base_forward(x)

        x, x_feat_after_aspp = self.head(c4)
        if self.aux:
            auxout = self.auxlayer(c3)
        if self.task == 'multi':
            if self.task_len == 3:
                x1, x_feat_after_aspp1 = self.head1(c4)
                x2, x_feat_after_aspp2 = self.head2(c4)
                x2 = x2 / torch.norm(x2, p=2, dim=1, keepdim=True)
                if self.aux:
                    auxout1 = self.auxlayer1(c3)
                    auxout2 = self.auxlayer2(c3)
                    auxout2 = auxout2 / torch.norm(auxout2, p=2, dim=1, keepdim=True)
                    return [[x, auxout, x_feat_after_aspp], [x1, auxout1, x_feat_after_aspp1], [x2, auxout2, x_feat_after_aspp2], c4]
                else:
                    return [[x, x_feat_after_aspp], [x1, x_feat_after_aspp1], [x2, x_feat_after_aspp2], c4]        
            else:       
                x1, x_feat_after_aspp1 = self.head1(c4)    
                if self.aux:
                    auxout1 = self.auxlayer1(c3)
                    return [[x, auxout, x_feat_after_aspp], [x1, auxout1, x_feat_after_aspp1], c4]
                else:
                    return [[x, x_feat_after_aspp], [x1, x_feat_after_aspp1], c4]
        else:
            return [x, auxout, x_feat_after_aspp]
class DeepLabV3_multi(SegBaseModel): 
    r"""DeepLabV3_multi

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).
    """

    def __init__(self, nclass, backbone='resnet50', aux=False, task_len=None,local_rank=None, pretrained_base=True, **kwargs):
        super(DeepLabV3_multi, self).__init__(nclass,aux, backbone,  task_len, local_rank, pretrained_base=pretrained_base, **kwargs)
        self.aux = aux
        self.task = 'multi'
        self.task_len = task_len
        
        if backbone == 'resnet18':
            in_channels = 512
        else:
            in_channels = 2048

        self.head = _DeepLabHead(in_channels, nclass, **kwargs)
        if self.aux:
            self.auxlayer = _FCNHead(in_channels // 2, nclass, **kwargs)
        if self.task=='multi':
            self.head1 = _DeepLabHead(in_channels, 1, **kwargs)
            self.head2 = _DeepLabHead(in_channels, 3, **kwargs)
            if self.aux:
                self.auxlayer1 = _FCNHead(in_channels // 2, 1, **kwargs)
                self.auxlayer2 = _FCNHead(in_channels // 2, 3, **kwargs)
        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def zero_grad_shared_modules(self):
            self.pretrained.zero_grad()

    def forward(self, x):
        size = x.size()[2:]
        # auxout, auxout1, auxout2 = torch.zeros(size), torch.zeros(size), torch.zeros(size)
        c1, c2, c3, c4 = self.base_forward(x)

        x, x_feat_after_aspp = self.head(c4)
        if self.aux:
            auxout = self.auxlayer(c3)
        if self.task == 'multi':
            if self.task_len == 3:
                x1, x_feat_after_aspp1 = self.head1(c4)
                x2, x_feat_after_aspp2 = self.head2(c4)
                x2 = x2 / torch.norm(x2, p=2, dim=1, keepdim=True)
                if self.aux:
                    auxout1 = self.auxlayer1(c3)
                    auxout2 = self.auxlayer2(c3)
                    auxout2 = auxout2 / torch.norm(auxout2, p=2, dim=1, keepdim=True)
                    return [[x, auxout, x_feat_after_aspp], [x1, auxout1, x_feat_after_aspp1], [x2, auxout2, x_feat_after_aspp2], c1,c2,c3,c4]
                else:
                    return [[x, x_feat_after_aspp], [x1, x_feat_after_aspp1], [x2, x_feat_after_aspp2], c4]        
            else:       
                x1, x_feat_after_aspp1 = self.head1(c4)    
                if self.aux:
                    auxout1 = self.auxlayer1(c3)
                    return [[x, auxout, x_feat_after_aspp], [x1, auxout1, x_feat_after_aspp1], c4]
                else:
                    return [[x, x_feat_after_aspp], [x1, x_feat_after_aspp1], c4]
        else:
            return [x, auxout, x_feat_after_aspp]
        
class DeepLabV3(SegBaseModel):
    r"""DeepLabV3

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).
    """

    def __init__(self, nclass, backbone='resnet50', aux=False, local_rank=None, pretrained_base=True, **kwargs):
        super(DeepLabV3, self).__init__(nclass,aux, backbone,  local_rank, pretrained_base=pretrained_base, **kwargs)
        self.aux = aux
        self.nclass = nclass
        if backbone == 'resnet18':
            in_channels = 512
        else:
            in_channels = 2048

        self.head = _DeepLabHead(in_channels, nclass, **kwargs)
        if self.aux:
            self.auxlayer = _FCNHead(in_channels // 2, nclass, **kwargs)
        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])
    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.base_forward(x)

        x, x_feat_after_aspp = self.head(c4)
        if self.nclass==3:
            x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        if self.aux:
            auxout = self.auxlayer(c3)
            if self.nclass == 3:
                auxout = auxout / torch.norm(auxout, p=2, dim=1, keepdim=True)
        return [x, auxout, x_feat_after_aspp, c1,c2,c3,c4]

class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight)
        #         # nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)
    def forward(self, x):
        return self.block(x)

class _DeepLabHead(nn.Module):
    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(in_channels, [12, 24, 36], norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        
        if in_channels == 512:
            out_channels = 128
        elif in_channels == 2048:
            out_channels = 256
        else:
            raise 

        self.block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, nclass, 1)
        )
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight)
        #         # nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.aspp(x)
        x = self.block[0:4](x)
        x_feat_after_aspp = x
        x = self.block[4](x)
        return x, x_feat_after_aspp

class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(_ASPP, self).__init__()
        if in_channels == 512:
            out_channels = 128
        elif in_channels == 2048:
            out_channels = 256
        else:
            raise 

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x


def get_deeplabv3(backbone='resnet50', local_rank=None, pretrained=None, 
                  pretrained_base=True, num_class=7, **kwargs):

    model = DeepLabV3(num_class, backbone=backbone, local_rank=local_rank, pretrained_base=pretrained_base, **kwargs)
    if pretrained != 'None':
        if local_rank is not None:
            device = torch.device(local_rank)
            model.load_state_dict(torch.load(pretrained, map_location=device))
        else:
            model.load_state_dict(torch.load(pretrained, map_location='cpu'))
            print('Load Checkpoint Model:{}_{}'.format(backbone, num_class))
    return model

def get_deeplabv3_multi(backbone='resnet50', local_rank=None, pretrained=None, 
                  pretrained_base=True, num_class=7, task_len=None,**kwargs):

    model = DeepLabV3_multi(num_class, backbone=backbone, task_len=task_len,local_rank=local_rank, pretrained_base=pretrained_base, **kwargs)
    if pretrained != 'None':
        if local_rank is not None:
            device = torch.device(local_rank)
            model.load_state_dict(torch.load(pretrained, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(pretrained, map_location='cpu'))
    return model

def get_resnet_multi(backbone='resnet50', local_rank=None, pretrained=None, 
                  pretrained_base=True, num_class=7, task_len=None,**kwargs):

    model = Resnet_multi(num_class, backbone=backbone, task_len=task_len,local_rank=local_rank, pretrained_base=pretrained_base, **kwargs)
    if pretrained != 'None':
        if local_rank is not None:
            device = torch.device(local_rank)
            model.load_state_dict(torch.load(pretrained, map_location=device))
    return model

def get_deeplabv3_mtan(backbone='resnet50', local_rank=None, pretrained=None, 
                  pretrained_base=True, num_class=7, **kwargs):

    model = DeepLabV3_mtan(num_class, backbone=backbone, local_rank=local_rank, pretrained_base=pretrained_base, **kwargs)
    if pretrained != 'None':
        if local_rank is not None:
            device = torch.device(local_rank)
            model.load_state_dict(torch.load(pretrained, map_location='cpu'))
    return model


if __name__ == '__main__':
    model = get_deeplabv3()
    img = torch.randn(2, 3, 480, 480)
    output = model(img)

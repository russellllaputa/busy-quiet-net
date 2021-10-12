import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from src.bpf import MotionBandPassFilter
from ops.depthwise_conv import *
from torch.nn.init import normal_, constant_
import math


__all__ = ['ResNet', 'resnet50', 'resnet101']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}

def shift(x, num_segments, fold_div=8):
    nt, c, h, w = x.size()
    n_batch = nt // num_segments
    x = x.view(n_batch, num_segments, c, h, w)

    fold = c // fold_div

    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
    out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

    return out.view(nt, c, h, w)

def depthwise_conv3x3(in_channels, stride=1):
    
    return DepthwiseConv2d(in_channels=in_channels, ks=[3, 3], stride=stride)

def depthwise_conv3x3x3(in_channels, stride=1):
    
    return DepthwiseConv3d(in_channels=in_channels, ks=[3, 3, 3], stride=[1,stride,stride])

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ShiftModule(nn.Module):
    """1D Temporal convolutions, the convs are initialized to act as the "Part shift" layer
    """
    def __init__(self, input_channels, num_segments, n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels
        self.num_segments = num_segments
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(2*self.fold, 2*self.fold, kernel_size=3, padding=1, groups=2*self.fold, bias=False)
        # weight_size: (2*self.fold, 1, 3)
        if mode == 'shift':
            # import pdb; pdb.set_trace()
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1 # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
            if 2*self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True
    def forward(self, x):
        # shift by conv
        # import pdb; pdb.set_trace()
        nt, c, h, w = x.size()
        n_batch = nt // self.num_segments
        x = x.view(n_batch, self.num_segments, c, h, w)
        x = x.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, num_segments)
        x = x.contiguous().view(n_batch*h*w, c, self.num_segments)
        x = self.conv(x)  # (n_batch*h*w, c, num_segments)
        x = x.view(n_batch, h, w, c, self.num_segments)
        x = x.permute([0, 4, 3, 1, 2])  # (n_batch, num_segments, c, h, w)
        x = x.contiguous().view(nt, c, h, w)
        return x

    
    
class BottleneckC(nn.Module):

    
    expansion = 4
    

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleneckC, self).__init__()
        
    
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.num_segments = num_segments
        self.width  = width
        self.num_features = planes*self.expansion


    def forward(self, x):
        
        identity = x
        out = shift(x, self.num_segments)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        
        return out

    
class BottleneckF(nn.Module):

    expansion = 4
    

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleneckF, self).__init__()
        
    
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.num_segments = num_segments
        self.width  = width
        self.num_features = planes*self.expansion
        
        if downsample is not None:
            self.downsample = nn.Sequential(
                            conv1x1(inplanes, planes * self.expansion, stride),
                            norm_layer(planes * self.expansion),
                        )



    def forward(self, x):
        
        identity = x
        out = shift(x, self.num_segments)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        
        return out


    
    
class DuoPath(nn.Module):

    
    expansion = 4
    

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, d_base_width=26, scale=4, lateral_c=False, d_stride=1, sigma=0.6, kernel_size=7):
        super(DuoPath, self).__init__()
        self.lateral_c = lateral_c

        self.path_f = BottleneckF(num_segments, inplanes, planes, stride, downsample=downsample, groups=groups,
                                  base_width=base_width, dilation=dilation, norm_layer=norm_layer) 

        self.path_c = BottleneckC(num_segments, inplanes, planes, stride, downsample=downsample, groups=groups,
                                  base_width=base_width, dilation=dilation, norm_layer=norm_layer)        
        if lateral_c:
            self.mbpf = MotionBandPassFilter(num_segments=num_segments, channels=planes*self.expansion, sigma=sigma, kernel_size=kernel_size, three_steps=False)
            self.d_up = nn.Sequential(
                conv1x1(planes*self.expansion, planes*self.expansion, stride=1),
                norm_layer(planes*self.expansion))
            self.d_down = nn.Sequential(
                conv1x1(planes*self.expansion, planes*self.expansion, stride=1),
                norm_layer(planes*self.expansion))


        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if lateral_c:
            nn.init.constant_(self.d_up[-1].weight, 0)
            nn.init.constant_(self.d_up[-1].bias, 0)
            nn.init.constant_(self.d_down[-1].weight, 0)
            nn.init.constant_(self.d_down[-1].bias, 0)

    def forward(self, x):
        x_c = x[0] # nt, c, h, w
        x_f = x[1]
        
        out_c = self.path_c(x_c)
        out_f = self.path_f(x_f)

        if self.lateral_c:
            fine1_st = self.mbpf(out_c)
            out_c = out_c + self.d_down(out_f) # nt, c1, h1, w1
            out_f = self.d_up(fine1_st) + out_f # nt, c2, h2, w2

        
        return (out_c, out_f)
    


class ResNet(nn.Module):

    def __init__(self, block, layers, data_length, num_segments, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, d_base_width=26, scale=4, dropout=0.5):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.num_segments = num_segments
        self.scale = scale
        self.data_length = data_length
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.mbpf1 = MotionBandPassFilter(num_segments=num_segments, channels=3, sigma=1.1, kernel_size=9, three_steps=True)
        
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        
        self.d_conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.d_bn1 = norm_layer(self.inplanes)
        
        
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(DuoPath, 64, layers[0], stride=1, d_stride=1, sigma=0.9, kernel_size=7, has_lc=True)
        self.layer2 = self._make_layer(DuoPath, 128, layers[1], stride=2, d_stride=2, sigma=0.9, kernel_size=7, has_lc=True)
        self.layer3 = self._make_layer(DuoPath, 256, layers[2], stride=2, d_stride=2, sigma=0.9, kernel_size=7, has_lc=True)
        self.layer4 = self._make_layer(DuoPath, 512, layers[3], stride=2, d_stride=2, sigma=0.9, kernel_size=3, has_lc=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        self.droput = nn.Dropout(p=dropout, inplace=True)



        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DuoBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                    
        std = 0.001
        normal_(self.fc.weight, 0, std)
        constant_(self.fc.bias, 0)
        
            
    def _make_layer(self, block, planes, blocks, stride=1, d_stride=1, sigma=0.6, kernel_size=7, dilate=False, has_lc=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        if has_lc:
            lateral_c1 = True
            lateral_c2 = True
        else:
            lateral_c1 = False
            lateral_c2 = False
        layers = []
        layers.append(block(self.num_segments, self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,
                            lateral_c=lateral_c1, d_stride=d_stride, sigma=sigma, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.num_segments, self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, lateral_c=lateral_c2, d_stride=1, sigma=sigma, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        t = self.num_segments
        nt, dc, h, w = x.size()
        d = self.data_length
        c = dc // d
        x = x.view(-1, t, d, c, h, w) # n, t, d, c, h, w
        x_c = x[:,:,0].reshape(-1, c, h, w) # nt, c, h, w
        x_f = self.mbpf1(x) # nt, c, h, w
        self.reps = x_f
        x_c = self.conv1(x_c)
        x_c = self.bn1(x_c)
        x_c = self.relu(x_c)
        x_c = self.maxpool(x_c)
        
        x_f = self.d_conv1(x_f)
        x_f = self.d_bn1(x_f)
        x_f = self.relu(x_f)
        x_f = self.maxpool(x_f)

        x = (x_c, x_f)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        out = torch.cat((x[0], x[1]), 0)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.droput(out) 
        out = self.fc(out)
        out = (out[:nt] + out[nt:]) / 2.
        

        return out

    def forward(self, x):
        return self._forward_impl(x)
    
    
def _resnet(arch, block, layers, pretrained, progress, data_length, num_segments, **kwargs):
    model = ResNet(block, layers, data_length, num_segments, **kwargs)
    if pretrained:
        print('loading pretrained weights...')
        import collections
        if arch == 'resnet50':
            pretrained_dict = torch.load('pretrained/cf_tworesnet_pretrained_weights.pth')
        elif arch == 'resnet101':
            pretrained_dict = torch.load('pretrained/fc_tworesnet101_pretrained_weights.pth')
        state_dict = model.state_dict()
        state_dict.update(pretrained_dict)
        model.load_state_dict(state_dict)

    return model


def resnet50(pretrained=False, progress=True, **kwargs):

    return _resnet('resnet50', DuoPath, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet101(pretrained=False, progress=True, **kwargs):

    return _resnet('resnet101', DuoPath, [3, 4, 23, 3], pretrained, progress, **kwargs)

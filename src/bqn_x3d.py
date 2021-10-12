# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any, Optional

import torch.nn as nn
from torch.hub import load_state_dict_from_url
from src.bpf import MotionBandPassFilter
from torch.nn.init import normal_, constant_

checkpoint_paths = {
    "x3d_xs": f"pretrained/X3D_XS_nofc.pth",
    "x3d_s": f"pretrained/X3D_S.pth",
    "x3d_m": f"pretrained/bq_x3dm.pth",
    "x3d_l": f"pretrained/bq_x3dm.pth",
}


def _x3d(
    model_num_class,
    pretrained: bool = True,
    progress: bool = True,
    checkpoint_path: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    model = create_x3d(model_num_class=model_num_class, **kwargs)
    if pretrained:
        print('loading pretrained weights...')
        pretrained_dict = torch.load(checkpoint_path)
        state_dict = model.state_dict()
        state_dict.update(pretrained_dict)
        model.load_state_dict(state_dict)
        print('loaded pretrained weights...')
    return model



def x3d_xs(
    model_num_class,
    pretrained=True,
    progress=True,
    input_clip_length=4,
    input_crop_size=160,
    **kwargs,
):
    r"""
    X3D-XS model architecture [1] trained on the Kinetics dataset.
    Model with pretrained weights has top1 accuracy of 69.12.
    [1] Christoph Feichtenhofer, "X3D: Expanding Architectures for
    Efficient Video Recognition." https://arxiv.org/abs/2004.04730
    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/x3d.py
    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _x3d(
        model_num_class=model_num_class,
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["x3d_xs"],
        input_clip_length=input_clip_length,
        input_crop_size=input_crop_size,
        **kwargs,
    )


def x3d_s(
    pretrained: bool = True,
    progress: bool = True,
    **kwargs,
):
    """
    X3D-XS model architecture [1] trained on the Kinetics dataset.
    Model with pretrained weights has top1 accuracy of 73.33.
    [1] Christoph Feichtenhofer, "X3D: Expanding Architectures for
    Efficient Video Recognition." https://arxiv.org/abs/2004.04730
    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/x3d.py
    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _x3d(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["x3d_s"],
        input_clip_length=13,
        input_crop_size=160,
        **kwargs,
    )


def x3d_m(
    model_num_class,
    pretrained=True,
    progress=True,
    input_clip_length=16,
    input_crop_size=256,
    **kwargs,
):
    """
    X3D-XS model architecture [1] trained on the Kinetics dataset.
    Model with pretrained weights has top1 accuracy of 75.94.
    [1] Christoph Feichtenhofer, "X3D: Expanding Architectures for
    Efficient Video Recognition." https://arxiv.org/abs/2004.04730
    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/x3d.py
    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _x3d(
        model_num_class=model_num_class,
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["x3d_m"],
        input_clip_length=input_clip_length,
        input_crop_size=input_crop_size,
        **kwargs,
    )


def x3d_l(
    model_num_class,
    pretrained=True,
    progress=True,
    input_clip_length=16,
    input_crop_size=256,
    **kwargs,
):
    """
    X3D-XS model architecture [1] trained on the Kinetics dataset.
    Model with pretrained weights has top1 accuracy of 77.44.
    [1] Christoph Feichtenhofer, "X3D: Expanding Architectures for
    Efficient Video Recognition." https://arxiv.org/abs/2004.04730
    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/x3d.py
    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _x3d(
        model_num_class=model_num_class,
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["x3d_l"],
        input_clip_length=input_clip_length,
        input_crop_size=input_crop_size,
        depth_factor=5.0,
        **kwargs,
    )


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorchvideo.layers.squeeze_excitation import SqueezeExcitation
from pytorchvideo.layers.convolutions import Conv2plus1d
from pytorchvideo.layers.swish import Swish
from pytorchvideo.layers.utils import round_repeats, round_width, set_attributes
from pytorchvideo.models.head import ResNetBasicHead
from pytorchvideo.models.resnet import BottleneckBlock, ResBlock, ResStage
from pytorchvideo.models.stem import ResNetBasicStem


def create_x3d_stem(
    *,
    # Conv configs.
    in_channels: int,
    out_channels: int,
    conv_kernel_size: Tuple[int] = (5, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    conv_padding: Tuple[int] = (2, 1, 1),
    # BN configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
) -> nn.Module:
    """
    Creates the stem layer for X3D. It performs spatial Conv, temporal Conv, BN, and Relu.
    ::
                                        Conv_xy
                                           ↓
                                        Conv_t
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
    Args:
        in_channels (int): input channel size of the convolution.
        out_channels (int): output channel size of the convolution.
        conv_kernel_size (tuple): convolutional kernel size(s).
        conv_stride (tuple): convolutional stride size(s).
        conv_padding (tuple): convolutional padding size(s).
        norm (callable): a callable that constructs normalization layer, options
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
        activation (callable): a callable that constructs activation layer, options
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).
    Returns:
        (nn.Module): X3D stem layer.
    """
    conv_xy_module = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(1, conv_kernel_size[1], conv_kernel_size[2]),
        stride=(1, conv_stride[1], conv_stride[2]),
        padding=(0, conv_padding[1], conv_padding[2]),
        bias=False,
    )
    conv_t_module = nn.Conv3d(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=(conv_kernel_size[0], 1, 1),
        stride=(conv_stride[0], 1, 1),
        padding=(conv_padding[0], 0, 0),
        bias=False,
        groups=out_channels,
    )
    stacked_conv_module = Conv2plus1d(
        conv_t=conv_xy_module,
        norm=None,
        activation=None,
        conv_xy=conv_t_module,
    )

    norm_module = (
        None
        if norm is None
        else norm(num_features=out_channels, eps=norm_eps, momentum=norm_momentum)
    )
    activation_module = None if activation is None else activation()

    return ResNetBasicStem(
        conv=stacked_conv_module,
        norm=norm_module,
        activation=activation_module,
        pool=None,
    )


class DuoStem(nn.Module):
    def __init__(self, *,
                 # Conv configs.
                 in_channels: int,
                 out_channels: int,
                 conv_kernel_size: Tuple[int] = (5, 3, 3),
                 conv_stride: Tuple[int] = (1, 2, 2),
                 conv_padding: Tuple[int] = (2, 1, 1),
                 # BN configs.
                 norm: Callable = nn.BatchNorm3d,
                 norm_eps: float = 1e-5,
                 norm_momentum: float = 0.1,
                 # Activation configs.
                 activation: Callable = nn.ReLU):
        super(DuoStem, self).__init__()

        self.path_f = create_x3d_stem(in_channels=in_channels,
                                      out_channels=out_channels,
                                      conv_kernel_size=conv_kernel_size,
                                      conv_stride=conv_stride,
                                      conv_padding=conv_padding,
                                      # BN configs.
                                      norm=norm,
                                      norm_eps=norm_eps,
                                      norm_momentum=norm_momentum,
                                      # Activation configs.
                                      activation=activation)
        
        self.path_c = create_x3d_stem(in_channels=in_channels,
                                      out_channels=out_channels,
                                      conv_kernel_size=conv_kernel_size,
                                      conv_stride=conv_stride,
                                      conv_padding=conv_padding,
                                      # BN configs.
                                      norm=norm,
                                      norm_eps=norm_eps,
                                      norm_momentum=norm_momentum,
                                      # Activation configs.
                                      activation=activation) 

    def forward(self, x):
        x_c = x[0] # n, c, t, h, w
        x_f = x[1]

        
        out_c = self.path_c(x_c)
        out_f = self.path_f(x_f)

        return (out_c, out_f)
    

def create_x3d_bottleneck_block(
    *,
    # Convolution configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    conv_kernel_size: Tuple[int] = (3, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    # Norm configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    se_ratio: float = 0.0625,
    # Activation configs.
    activation: Callable = nn.ReLU,
    inner_act: Callable = Swish,
) -> nn.Module:
    """
    Bottleneck block for X3D: a sequence of Conv, Normalization with optional SE block,
    and Activations repeated in the following order:
    ::
                                    Conv3d (conv_a)
                                           ↓
                                 Normalization (norm_a)
                                           ↓
                                   Activation (act_a)
                                           ↓
                                    Conv3d (conv_b)
                                           ↓
                                 Normalization (norm_b)
                                           ↓
                                 Squeeze-and-Excitation
                                           ↓
                                   Activation (act_b)
                                           ↓
                                    Conv3d (conv_c)
                                           ↓
                                 Normalization (norm_c)
    Args:
        dim_in (int): input channel size to the bottleneck block.
        dim_inner (int): intermediate channel size of the bottleneck.
        dim_out (int): output channel size of the bottleneck.
        conv_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        conv_stride (tuple): convolutional stride size(s) for conv_b.
        norm (callable): a callable that constructs normalization layer, examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
        se_ratio (float): if > 0, apply SE to the 3x3x3 conv, with the SE
            channel dimensionality being se_ratio times the 3x3x3 conv dim.
        activation (callable): a callable that constructs activation layer, examples
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).
        inner_act (callable): whether use Swish activation for act_b or not.
    Returns:
        (nn.Module): X3D bottleneck block.
    """
    # 1x1x1 Conv
    conv_a = nn.Conv3d(
        in_channels=dim_in, out_channels=dim_inner, kernel_size=(1, 1, 1), bias=False
    )
    norm_a = (
        None
        if norm is None
        else norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
    )
    act_a = None if activation is None else activation()

    # 3x3x3 Conv
    conv_b = nn.Conv3d(
        in_channels=dim_inner,
        out_channels=dim_inner,
        kernel_size=conv_kernel_size,
        stride=conv_stride,
        padding=[size // 2 for size in conv_kernel_size],
        bias=False,
        groups=dim_inner,
        dilation=(1, 1, 1),
    )
    se = (
        SqueezeExcitation(
            num_channels=dim_inner,
            num_channels_reduced=round_width(dim_inner, se_ratio),
            is_3d=True,
        )
        if se_ratio > 0.0
        else nn.Identity()
    )
    norm_b = nn.Sequential(
        (
            nn.Identity()
            if norm is None
            else norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
        ),
        se,
    )
    act_b = None if inner_act is None else inner_act()

    # 1x1x1 Conv
    conv_c = nn.Conv3d(
        in_channels=dim_inner, out_channels=dim_out, kernel_size=(1, 1, 1), bias=False
    )
    norm_c = (
        None
        if norm is None
        else norm(num_features=dim_out, eps=norm_eps, momentum=norm_momentum)
    )

    return BottleneckBlock(
        conv_a=conv_a,
        norm_a=norm_a,
        act_a=act_a,
        conv_b=conv_b,
        norm_b=norm_b,
        act_b=act_b,
        conv_c=conv_c,
        norm_c=norm_c,
    )


def create_x3d_res_block(
    *,
    # Bottleneck Block configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    bottleneck: Callable = create_x3d_bottleneck_block,
    use_shortcut: bool = True,
    # Conv configs.
    conv_kernel_size: Tuple[int] = (3, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    # Norm configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    se_ratio: float = 0.0625,
    # Activation configs.
    activation: Callable = nn.ReLU,
    inner_act: Callable = Swish,
) -> nn.Module:
    """
    Residual block for X3D. Performs a summation between an identity shortcut in branch1 and a
    main block in branch2. When the input and output dimensions are different, a
    convolution followed by a normalization will be performed.
    ::
                                         Input
                                           |-------+
                                           ↓       |
                                         Block     |
                                           ↓       |
                                       Summation ←-+
                                           ↓
                                       Activation
    Args:
        dim_in (int): input channel size to the bottleneck block.
        dim_inner (int): intermediate channel size of the bottleneck.
        dim_out (int): output channel size of the bottleneck.
        bottleneck (callable): a callable for create_x3d_bottleneck_block.
        conv_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        conv_stride (tuple): convolutional stride size(s) for conv_b.
        norm (callable): a callable that constructs normalization layer, examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
        se_ratio (float): if > 0, apply SE to the 3x3x3 conv, with the SE
            channel dimensionality being se_ratio times the 3x3x3 conv dim.
        activation (callable): a callable that constructs activation layer, examples
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).
        inner_act (callable): whether use Swish activation for act_b or not.
    Returns:
        (nn.Module): X3D block layer.
    """

    norm_model = None
    if norm is not None and dim_in != dim_out:
        norm_model = norm(num_features=dim_out)

    return ResBlock(
        branch1_conv=nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=conv_stride,
            bias=False,
        )
        if (dim_in != dim_out or np.prod(conv_stride) > 1) and use_shortcut
        else None,
        branch1_norm=norm_model if dim_in != dim_out and use_shortcut else None,
        branch2=bottleneck(
            dim_in=dim_in,
            dim_inner=dim_inner,
            dim_out=dim_out,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            se_ratio=se_ratio,
            activation=activation,
            inner_act=inner_act,
        ),
        activation=None if activation is None else activation(),
        branch_fusion=lambda x, y: x + y,
    )

class DuoPath(nn.Module):


    def __init__(self, *,
                 num_segments: int,
                 # Bottleneck Block configs.
                 dim_in: int,
                 dim_inner: int,
                 dim_out: int,
                 bottleneck: Callable = create_x3d_bottleneck_block,
                 use_shortcut: bool = True,
                 # Conv configs.
                 conv_kernel_size: Tuple[int] = (3, 3, 3),
                 conv_stride: Tuple[int] = (1, 2, 2),
                 # Norm configs.
                 norm: Callable = nn.BatchNorm3d,
                 norm_eps: float = 1e-5,
                 norm_momentum: float = 0.1,
                 se_ratio: float = 0.0625,
                 # Activation configs.
                 activation: Callable = nn.ReLU,
                 inner_act: Callable = Swish,
                 sigma=0.9, kernel_size=7, c_scale=160./256.):
        super(DuoPath, self).__init__()

        self.path_f = create_x3d_res_block(dim_in=dim_in, 
                                           dim_inner=dim_inner, 
                                           dim_out=dim_out, 
                                           bottleneck=bottleneck, 
                                           conv_kernel_size=conv_kernel_size,
                                           conv_stride=conv_stride,
                                           norm=norm,
                                           norm_eps=norm_eps,
                                           norm_momentum=norm_momentum,
                                           se_ratio=se_ratio,
                                           activation=activation,
                                           inner_act=inner_act)

        self.path_c = create_x3d_res_block(dim_in=dim_in, 
                                           dim_inner=dim_inner, 
                                           dim_out=dim_out, 
                                           bottleneck=bottleneck, 
                                           conv_kernel_size=conv_kernel_size,
                                           conv_stride=conv_stride,
                                           norm=norm,
                                           norm_eps=norm_eps,
                                           norm_momentum=norm_momentum,
                                           se_ratio=se_ratio,
                                           activation=activation,
                                           inner_act=inner_act)
        self.mbpf = MotionBandPassFilter(num_segments=num_segments, channels=dim_out, sigma=sigma, kernel_size=kernel_size, three_steps=False)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=1./c_scale, mode='bilinear', align_corners=False),
            nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(dim_out))
        
        self.down = nn.Sequential(
            nn.Upsample(scale_factor=c_scale, mode='bilinear', align_corners=False),
            nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(dim_out))


        nn.init.constant_(self.up[-1].weight, 0)
        nn.init.constant_(self.up[-1].bias, 0)

        nn.init.constant_(self.down[-1].weight, 0)
        nn.init.constant_(self.down[-1].bias, 0)

    def forward(self, x):
        x_c = x[0] # n, c, t, h, w
        x_f = x[1]

                
        out_c = self.path_c(x_c)
        out_f = self.path_f(x_f)

        n1, c1, t1, h1, w1 = out_c.size()
        n2, c2, t2, h2, w2 = out_f.size()
        
        out_c = out_c.transpose(1, 2) #n, t, c, h, w
        out_c = out_c.reshape(-1, c1, h1, w1) #nt, c, h, w

        out_f = out_f.transpose(1, 2) #n, t, c, h, w
        out_f = out_f.reshape(-1, c2, h2, w2) #nt, c, h, w
        
        out_f = self.up(self.mbpf(out_c)) + out_f # nt, c2, h2, w2
        out_c = out_c + self.down(out_f) # nt, c1, h1, w1
            
        out_c = out_c.view(n1, t1, c1, h1, w1).transpose(1,2).reshape(n1, c1, t1, h1, w1) # n, c1, t, h1, w1
        out_f = out_f.view(n2, t2, c2, h2, w2).transpose(1,2).reshape(n2, c2, t2, h2, w2)

            
        return (out_c, out_f)
    

def create_x3d_res_stage(
    *,
    num_segments: int,
    # Stage configs.
    depth: int,
    # Bottleneck Block configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    bottleneck: Callable = create_x3d_bottleneck_block,
    # Conv configs.
    conv_kernel_size: Tuple[int] = (3, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    # Norm configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    se_ratio: float = 0.0625,
    # Activation configs.
    activation: Callable = nn.ReLU,
    inner_act: Callable = Swish,
) -> nn.Module:
    """
    Create Residual Stage, which composes sequential blocks that make up X3D.
    ::
                                        Input
                                           ↓
                                       ResBlock
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                       ResBlock
    Args:
        depth (init): number of blocks to create.
        dim_in (int): input channel size to the bottleneck block.
        dim_inner (int): intermediate channel size of the bottleneck.
        dim_out (int): output channel size of the bottleneck.
        bottleneck (callable): a callable for create_x3d_bottleneck_block.
        conv_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        conv_stride (tuple): convolutional stride size(s) for conv_b.
        norm (callable): a callable that constructs normalization layer, examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
        se_ratio (float): if > 0, apply SE to the 3x3x3 conv, with the SE
            channel dimensionality being se_ratio times the 3x3x3 conv dim.
        activation (callable): a callable that constructs activation layer, examples
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).
        inner_act (callable): whether use Swish activation for act_b or not.
    Returns:
        (nn.Module): X3D stage layer.
    """
    res_blocks = []
    for idx in range(depth):
        block = DuoPath(
            num_segments= num_segments,
            dim_in=dim_in if idx == 0 else dim_out,
            dim_inner=dim_inner,
            dim_out=dim_out,
            bottleneck=bottleneck,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride if idx == 0 else (1, 1, 1),
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            se_ratio=(se_ratio if (idx + 1) % 2 else 0.0),
            activation=activation,
            inner_act=inner_act,
            sigma=0.9,
            kernel_size=7 if dim_out < 192 else 3,
            c_scale=160./256.
        )
        res_blocks.append(block)

    return ResStage(res_blocks=nn.ModuleList(res_blocks))


def create_x3d_head(
    *,
    # Projection configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    num_classes: int,
    # Pooling configs.
    pool_act: Callable = nn.ReLU,
    pool_kernel_size: Tuple[int] = (13, 5, 5),
    # BN configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    bn_lin5_on=False,
    # Dropout configs.
    dropout_rate: float = 0.5,
    # Activation configs.
    activation: Callable = None,
    # Output configs.
    output_with_global_average: bool = True,
) -> nn.Module:
    """
    Creates X3D head. This layer performs an projected pooling operation followed
    by an dropout, a fully-connected projection, an activation layer and a global
    spatiotemporal averaging.
    ::
                                     ProjectedPool
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation
                                           ↓
                                       Averaging
    Args:
        dim_in (int): input channel size of the X3D head.
        dim_inner (int): intermediate channel size of the X3D head.
        dim_out (int): output channel size of the X3D head.
        num_classes (int): the number of classes for the video dataset.
        pool_act (callable): a callable that constructs resnet pool activation
            layer such as nn.ReLU.
        pool_kernel_size (tuple): pooling kernel size(s) when not using adaptive
            pooling.
        norm (callable): a callable that constructs normalization layer, examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
        bn_lin5_on (bool): if True, perform normalization on the features
            before the classifier.
        dropout_rate (float): dropout rate.
        activation (callable): a callable that constructs resnet head activation
            layer, examples include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not
            applying activation).
        output_with_global_average (bool): if True, perform global averaging on temporal
            and spatial dimensions and reshape output to batch_size x out_features.
    Returns:
        (nn.Module): X3D head layer.
    """
    pre_conv_module = nn.Conv3d(
        in_channels=dim_in, out_channels=dim_inner, kernel_size=(1, 1, 1), bias=False
    )

    pre_norm_module = norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
    pre_act_module = None if pool_act is None else pool_act()

    if pool_kernel_size is None:
        pool_module = nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        pool_module = nn.AvgPool3d(pool_kernel_size, stride=1)

    post_conv_module = nn.Conv3d(
        in_channels=dim_inner, out_channels=dim_out, kernel_size=(1, 1, 1), bias=False
    )

    if bn_lin5_on:
        post_norm_module = norm(
            num_features=dim_out, eps=norm_eps, momentum=norm_momentum
        )
    else:
        post_norm_module = None
    post_act_module = None if pool_act is None else pool_act()

    projected_pool_module = ProjectedPool(
        pre_conv=pre_conv_module,
        pre_norm=pre_norm_module,
        pre_act=pre_act_module,
        pool=pool_module,
        post_conv=post_conv_module,
        post_norm=post_norm_module,
        post_act=post_act_module,
    )

    if activation is None:
        activation_module = None
    elif activation == nn.Softmax:
        activation_module = activation(dim=1)
    elif activation == nn.Sigmoid:
        activation_module = activation()
    else:
        raise NotImplementedError(
            "{} is not supported as an activation" "function.".format(activation)
        )

    if output_with_global_average:
        output_pool = nn.AdaptiveAvgPool3d(1)
    else:
        output_pool = None

    return ResNetBasicHead(
        proj=nn.Linear(dim_out, num_classes, bias=True),
        activation=activation_module,
        pool=projected_pool_module,
        dropout=nn.Dropout(dropout_rate) if dropout_rate > 0 else None,
        output_pool=output_pool,
    )



class DuoHead(nn.Module):
    def __init__(self, *,
                    # Projection configs.
                    dim_in: int,
                    dim_inner: int,
                    dim_out: int,
                    num_classes: int,
                    # Pooling configs.
                    pool_act: Callable = nn.ReLU,
                    pool_kernel_size: Tuple[int] = (13, 5, 5),
                    # BN configs.
                    norm: Callable = nn.BatchNorm3d,
                    norm_eps: float = 1e-5,
                    norm_momentum: float = 0.1,
                    bn_lin5_on=False,
                    # Dropout configs.
                    dropout_rate: float = 0.5,
                    # Activation configs.
                    activation: Callable = None,
                    # Output configs.
                    output_with_global_average: bool = True,):
        super(DuoHead, self).__init__()

        self.path_f = create_x3d_head(dim_in=dim_in,
                                      dim_inner=dim_inner,
                                      dim_out=dim_out,
                                      num_classes=num_classes,
                                      # Pooling configs.
                                      pool_act=pool_act,
                                      pool_kernel_size=pool_kernel_size,
                                      # BN configs.
                                      norm=norm,
                                      norm_eps=norm_eps,
                                      norm_momentum=norm_momentum,
                                      bn_lin5_on=bn_lin5_on,
                                      # Dropout configs.
                                      dropout_rate=dropout_rate,
                                      # Activation configs.
                                      activation=activation,
                                      # Output configs.
                                      output_with_global_average=output_with_global_average,)
        
        self.path_c = create_x3d_head(dim_in=dim_in,
                                      dim_inner=dim_inner,
                                      dim_out=dim_out,
                                      num_classes=num_classes,
                                      # Pooling configs.
                                      pool_act=pool_act,
                                      pool_kernel_size=pool_kernel_size,
                                      # BN configs.
                                      norm=norm,
                                      norm_eps=norm_eps,
                                      norm_momentum=norm_momentum,
                                      bn_lin5_on=bn_lin5_on,
                                      # Dropout configs.
                                      dropout_rate=dropout_rate,
                                      # Activation configs.
                                      activation=activation,
                                      # Output configs.
                                      output_with_global_average=output_with_global_average,)

    def forward(self, x):
        x_c = x[0] # n, c, t, h, w
        x_f = x[1]

        
        out_c = self.path_c(x_c)
        out_f = self.path_f(x_f)

        return (out_c + out_f) / 2.
    
def create_x3d(
    *,
    # Input clip configs.
    input_channel: int = 3,
    input_clip_length: int = 13,
    input_crop_size: int = 160,
    # Model configs.
    model_num_class: int = 400,
    dropout_rate: float = 0.5,
    width_factor: float = 2.0,
    depth_factor: float = 2.2,
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem configs.
    stem_dim_in: int = 12,
    stem_conv_kernel_size: Tuple[int] = (5, 3, 3),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    # Stage configs.
    stage_conv_kernel_size: Tuple[Tuple[int]] = (
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
    ),
    stage_spatial_stride: Tuple[int] = (2, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 1, 1, 1),
    bottleneck: Callable = create_x3d_bottleneck_block,
    bottleneck_factor: float = 2.25,
    se_ratio: float = 0.0625,
    inner_act: Callable = Swish,
    # Head configs.
    head_dim_out: int = 2048,
    head_pool_act: Callable = nn.ReLU,
    head_bn_lin5_on: bool = False,
    head_activation: Callable = None,
    head_output_with_global_average: bool = True,
) -> nn.Module:
    """
    X3D model builder. It builds a X3D network backbone, which is a ResNet.
    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    ::
                                         Input
                                           ↓
                                         Stem
                                           ↓
                                         Stage 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Stage N
                                           ↓
                                         Head
    Args:
        input_channel (int): number of channels for the input video clip.
        input_clip_length (int): length of the input video clip. Value for
            different models: X3D-XS: 4; X3D-S: 13; X3D-M: 16; X3D-L: 16.
        input_crop_size (int): spatial resolution of the input video clip.
            Value for different models: X3D-XS: 160; X3D-S: 160; X3D-M: 224;
            X3D-L: 312.
        model_num_class (int): the number of classes for the video dataset.
        dropout_rate (float): dropout rate.
        width_factor (float): width expansion factor.
        depth_factor (float): depth expansion factor. Value for different
            models: X3D-XS: 2.2; X3D-S: 2.2; X3D-M: 2.2; X3D-L: 5.0.
        norm (callable): a callable that constructs normalization layer.
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
        activation (callable): a callable that constructs activation layer.
        stem_dim_in (int): input channel size for stem before expansion.
        stem_conv_kernel_size (tuple): convolutional kernel size(s) of stem.
        stem_conv_stride (tuple): convolutional stride size(s) of stem.
        stage_conv_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        stage_spatial_stride (tuple): the spatial stride for each stage.
        stage_temporal_stride (tuple): the temporal stride for each stage.
        bottleneck_factor (float): bottleneck expansion factor for the 3x3x3 conv.
        se_ratio (float): if > 0, apply SE to the 3x3x3 conv, with the SE
            channel dimensionality being se_ratio times the 3x3x3 conv dim.
        inner_act (callable): whether use Swish activation for act_b or not.
        head_dim_out (int): output channel size of the X3D head.
        head_pool_act (callable): a callable that constructs resnet pool activation
            layer such as nn.ReLU.
        head_bn_lin5_on (bool): if True, perform normalization on the features
            before the classifier.
        head_activation (callable): a callable that constructs activation layer.
        head_output_with_global_average (bool): if True, perform global averaging on
            the head output.
    Returns:
        (nn.Module): the X3D network.
    """

    torch._C._log_api_usage_once("PYTORCHVIDEO.model.create_x3d")

    blocks = []
    # Create stem for X3D.
    stem_dim_out = round_width(stem_dim_in, width_factor)
    stem = DuoStem(
        in_channels=input_channel,
        out_channels=stem_dim_out,
        conv_kernel_size=stem_conv_kernel_size,
        conv_stride=stem_conv_stride,
        conv_padding=[size // 2 for size in stem_conv_kernel_size],
        norm=norm,
        norm_eps=norm_eps,
        norm_momentum=norm_momentum,
        activation=activation,
    )
    blocks.append(stem)

    # Compute the depth and dimension for each stage
    stage_depths = [1, 2, 5, 3]
    exp_stage = 2.0
    stage_dim1 = stem_dim_in
    stage_dim2 = round_width(stage_dim1, exp_stage, divisor=8)
    stage_dim3 = round_width(stage_dim2, exp_stage, divisor=8)
    stage_dim4 = round_width(stage_dim3, exp_stage, divisor=8)
    stage_dims = [stage_dim1, stage_dim2, stage_dim3, stage_dim4]

    dim_in = stem_dim_out
    # Create each stage for X3D.
    for idx in range(len(stage_depths)):
        dim_out = round_width(stage_dims[idx], width_factor)
        dim_inner = int(bottleneck_factor * dim_out)
        depth = round_repeats(stage_depths[idx], depth_factor)

        stage_conv_stride = (
            stage_temporal_stride[idx],
            stage_spatial_stride[idx],
            stage_spatial_stride[idx],
        )

        stage = create_x3d_res_stage(
            num_segments=input_clip_length,
            depth=depth,
            dim_in=dim_in,
            dim_inner=dim_inner,
            dim_out=dim_out,
            bottleneck=bottleneck,
            conv_kernel_size=stage_conv_kernel_size[idx],
            conv_stride=stage_conv_stride,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            se_ratio=se_ratio,
            activation=activation,
            inner_act=inner_act,
        )
        blocks.append(stage)
        dim_in = dim_out

    # Create head for X3D.
    total_spatial_stride = stem_conv_stride[1] * np.prod(stage_spatial_stride)
    total_temporal_stride = stem_conv_stride[0] * np.prod(stage_temporal_stride)

    assert (
        input_clip_length >= total_temporal_stride
    ), "Clip length doesn't match temporal stride!"
    assert (
        input_crop_size >= total_spatial_stride
    ), "Crop size doesn't match spatial stride!"

    head_pool_kernel_size = (
        input_clip_length // total_temporal_stride,
        int(math.ceil(input_crop_size / total_spatial_stride)),
        int(math.ceil(input_crop_size / total_spatial_stride)),
    )

    head = DuoHead(
        dim_in=dim_out,
        dim_inner=dim_inner,
        dim_out=head_dim_out,
        num_classes=model_num_class,
        pool_act=head_pool_act,
        pool_kernel_size=None,
        norm=norm,
        norm_eps=norm_eps,
        norm_momentum=norm_momentum,
        bn_lin5_on=head_bn_lin5_on,
        dropout_rate=dropout_rate,
        activation=head_activation,
        output_with_global_average=head_output_with_global_average,
    )
    blocks.append(head)
    return Net(blocks=nn.ModuleList(blocks), num_segments=input_clip_length)


class ProjectedPool(nn.Module):
    """
    A pooling module augmented with Conv, Normalization and Activation both
    before and after pooling for the head layer of X3D.
    ::
                                    Conv3d (pre_conv)
                                           ↓
                                 Normalization (pre_norm)
                                           ↓
                                   Activation (pre_act)
                                           ↓
                                        Pool3d
                                           ↓
                                    Conv3d (post_conv)
                                           ↓
                                 Normalization (post_norm)
                                           ↓
                                   Activation (post_act)
    """

    def __init__(
        self,
        *,
        pre_conv: nn.Module = None,
        pre_norm: nn.Module = None,
        pre_act: nn.Module = None,
        pool: nn.Module = None,
        post_conv: nn.Module = None,
        post_norm: nn.Module = None,
        post_act: nn.Module = None,
    ) -> None:
        """
        Args:
            pre_conv (torch.nn.modules): convolutional module.
            pre_norm (torch.nn.modules): normalization module.
            pre_act (torch.nn.modules): activation module.
            pool (torch.nn.modules): pooling module.
            post_conv (torch.nn.modules): convolutional module.
            post_norm (torch.nn.modules): normalization module.
            post_act (torch.nn.modules): activation module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.pre_conv is not None
        assert self.pool is not None
        assert self.post_conv is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_conv(x)

        if self.pre_norm is not None:
            x = self.pre_norm(x)
        if self.pre_act is not None:
            x = self.pre_act(x)

        x = self.pool(x)
        x = self.post_conv(x)

        if self.post_norm is not None:
            x = self.post_norm(x)
        if self.post_act is not None:
            x = self.post_act(x)
        return x
    
    
    
    
from typing import List, Optional

import torch
import torch.nn as nn
from pytorchvideo.layers.utils import set_attributes
from pytorchvideo.models.weight_init import init_net_weights


class Net(nn.Module):
    """
    Build a general Net models with a list of blocks for video recognition.
    ::
                                         Input
                                           ↓
                                         Block 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Block N
                                           ↓
    The ResNet builder can be found in `create_resnet`.
    """

    def __init__(self, *, num_segments, blocks: nn.ModuleList) -> None:
        """
        Args:
            blocks (torch.nn.module_list): the list of block modules.
        """
        super().__init__()
        assert blocks is not None
        self.blocks = blocks
        init_net_weights(self)
        self.num_segments = num_segments
        self.mbpf1 = MotionBandPassFilter(num_segments=num_segments, channels=3, sigma=1.1, kernel_size=9, three_steps=True)
        self.down = nn.Upsample(scale_factor=160./256., mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.num_segments
        nt, dc, h, w = x.size()
        x = x.view(-1, t, 3, 3, h, w) # n, t, d, c, h, w
        
        x_c = x[:,:,0].reshape(-1, 3, h, w) # nt, c, h, w
        x_c = self.down(x_c)
        x_c = x_c.view(-1, t, 3, x_c.size(-2), x_c.size(-1)).transpose(1,2)
        
        x_f = self.mbpf1(x)
        x_f = x_f.view(-1, t, 3, x_f.size(-2), x_f.size(-1)).transpose(1,2) # n, c, t, h, w
        x = (x_c, x_f)
        for idx in range(len(self.blocks)):
            x = self.blocks[idx](x)
        return x
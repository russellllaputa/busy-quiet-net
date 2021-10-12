# from __future__ import absolute_import
from torch import nn
from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_
from ops.depthwise_conv import DiagonalwiseRefactorization
from src.bpf import *

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, data_length, arch, input_size=224, dropout=0.5, partial_bn=True, print_spec=True, fc_lr5=True, modality='HP'):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_class = num_class
        self.num_segments = num_segments
        self.fc_lr5 = fc_lr5
        self.dropout = dropout
        self.data_length = data_length
        self.arch = arch
        self.input_size = input_size
        if 'x3d' not in self.arch:
            self.consensus = ConsensusModule('avg')
            
        self._prepare_base_model()
    
#         if self.modality == 'HP':
#             print("Converting the ImageNet model to a mbpf init model")
#             self.base_model = self._construct_mbpf_model(self.base_model)
#             print("Done. Flow model ready...")
            
        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)
        

    def _prepare_base_model(self):
        if self.modality in ['RGB', 'Flow', 'RGBDiff']:
            from src import resnet as base_arch
        elif self.modality == 'HP':
            from src import fine_net as base_arch
        if self.arch == 'resnet50':
            self.base_model = base_arch.resnet50(pretrained=True, num_classes=self.num_class, dropout=self.dropout, data_length=self.data_length, num_segments=self.num_segments)
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
        if self.arch == 'resnet101':
            self.base_model = base_arch.resnet101(pretrained=True, num_classes=self.num_class, dropout=self.dropout, data_length=self.data_length, num_segments=self.num_segments)
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
        elif self.arch == 'x3dm':
            if self.modality == 'HP':
                from src.busy_x3d import x3d_m
                self.base_model = x3d_m(model_num_class=self.num_class, pretrained=True, progress=True, input_clip_length=self.num_segments, input_crop_size=self.input_size)
            else:
                from src.x3d import x3d_m
                self.base_model = x3d_m(model_num_class=self.num_class, pretrained=True, progress=True, input_clip_length=self.num_segments, input_crop_size=self.input_size)
            self.input_mean = [0.45, 0.45, 0.45]
            self.input_std = [0.225, 0.225, 0.225]
            


        if self.modality == 'Flow':
            self.input_mean = [0.5]
            self.input_std = [np.mean(self.input_std)]
        elif self.modality == 'RGBDiff':
            self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.data_length
            self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.data_length

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        nl_weight = []
        nl_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for name, m in self.named_modules():
            if isinstance(m, (LoG ,LO)):
                ps = list(m.parameters())
                lr5_weight.append(ps[0])
                if len(ps) == 2:
                    lr10_bias.append(ps[1])
            elif 'nl' in name and isinstance(m, (nn.Conv1d ,nn.Conv2d, nn.Conv3d, DiagonalwiseRefactorization, nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                ps = list(m.parameters())
                nl_weight.append(ps[0])
                if len(ps) == 2:
                    nl_bias.append(ps[1])
                
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, DiagonalwiseRefactorization)):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif ('fc' in name or 'proj' in name) and isinstance(m, (torch.nn.Linear, nn.BatchNorm1d, nn.SyncBatchNorm)):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality in ['Flow'] else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality in ['Flow'] else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for nl
            {'params': nl_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "nl_weight"},
            {'params': nl_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "nl_bias"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]
    
    def forward(self, input):
        #n*t*data_length, c, h, w
        num_channels = 3 if self.modality in ["RGB", 'HP', 'RGBDiff'] else 2 
        num_channels *= self.data_length
        #nt, dc, h, w
        x = input.view((-1, num_channels) + input.size()[-2:])
        output = self.base_model(x)
        
        if 'x3d' not in self.arch:
            output = output.view((-1, self.num_segments) + output.size()[1:])
            output = self.consensus(output)        
            return output.squeeze(1)
        else:
            return output
    
    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if self.modality == 'RGB' or self.modality == 'HP':
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                print('=> NO FLIP!!!')
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])

    def _construct_flow_model(self, base_model):
        # modify the first convolution layers
        conv_layer = base_model.conv1
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2*self.data_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv = nn.Conv2d(2, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        base_model.conv1 = new_conv


        return base_model
    
    def _construct_mbpf_model(self, base_model):
        # modify the first convolution layers
        conv_layer = base_model.conv1
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3*(self.data_length-2), ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv = nn.Conv2d(2, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        base_model.conv1 = new_conv


        return base_model
from __future__ import print_function, division
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from ops.utils import expand_dim


def transform_trans_out(trans_out1):
#     b = trans_out1.size(0)
#     repl = np.zeros([b,2,3])
#     repl[:,0,0]=1
#     repl[:,1,1]=1 
#     trans_out1 = torch.Tensor(repl).cuda()

    trans_out1 = trans_out1.view(-1, 3)

    trans_out1_theta = trans_out1[:, 2]
    trans_out1_2 = trans_out1[:, 0].unsqueeze(1)
    trans_out1_5 = trans_out1[:, 1].unsqueeze(1)
    trans_out1_0 = 1. / 2.* torch.cos(trans_out1_theta).unsqueeze(1)
    trans_out1_1 = - 1. / 2. * torch.sin(trans_out1_theta).unsqueeze(1)
    trans_out1_3 = 1. / 2 * torch.sin(trans_out1_theta).unsqueeze(1)
    trans_out1_4 = 1. / 2 * torch.cos(trans_out1_theta).unsqueeze(1)
    trans_out1 = torch.cat((trans_out1_0, trans_out1_1, trans_out1_2, trans_out1_3, trans_out1_4, trans_out1_5), dim=1)

    trans_out1 = trans_out1.view(-1, 2, 3)
#     print("trans_out1:{}".format(trans_out1))

    return trans_out1


class GeometricTnfAffine(object):
    """
    Geometric transfromation to an image batch (wrapped in a PyTorch Variable)
    ( can be used with no transformation to perform bilinear resizing )
    """
    def __init__(self, out_h=240, out_w=240, offset_factor=None):
        self.out_h = out_h
        self.out_w = out_w
        self.offset_factor = offset_factor

        self.gridGen = AffineGridGenV3(out_h=out_h, out_w=out_w)

        if offset_factor is not None:
            self.gridGen.grid_X=self.gridGen.grid_X/offset_factor
            self.gridGen.grid_Y=self.gridGen.grid_Y/offset_factor

    def __call__(self, image_batch, theta_batch=None, out_h=None, out_w=None, return_warped_image=True, return_sampling_grid=False, padding_factor=1.0, crop_factor=1.0):

        sampling_grid = self.gridGen(theta_batch)

        # rescale grid according to crop_factor and padding_factor
        if padding_factor != 1 or crop_factor !=1:
            sampling_grid = sampling_grid*(padding_factor*crop_factor)
        # rescale grid according to offset_factor
        if self.offset_factor is not None:
            sampling_grid = sampling_grid*self.offset_factor

        if return_sampling_grid and not return_warped_image:
            return sampling_grid

        # sample transformed image
        warped_image_batch = F.grid_sample(image_batch, sampling_grid, align_corners=True)

        if return_sampling_grid and return_warped_image:
            return (warped_image_batch, sampling_grid)

        return warped_image_batch
    

class AffineGridGenV3(Module):
    def __init__(self, out_h=240, out_w=240):
        super(AffineGridGenV3, self).__init__()
        self.out_h, self.out_w = out_h, out_w

        # create grid in numpy
        self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))

    def forward(self, theta):
        b=theta.size(0)
        if not theta.size()==(b,6):
            theta = theta.view(b,6)
            theta = theta.contiguous()

        grid_X = torch.Tensor(self.grid_X).unsqueeze(0).unsqueeze(3).cuda()
        grid_Y = torch.Tensor(self.grid_Y).unsqueeze(0).unsqueeze(3).cuda()
        grid_X = Variable(grid_X,requires_grad=False)
        grid_Y = Variable(grid_Y,requires_grad=False)

        t0=theta[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t1=theta[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t2=theta[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t3=theta[:,3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t4=theta[:,4].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t5=theta[:,5].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        X = expand_dim(grid_X,0,b)
        Y = expand_dim(grid_Y,0,b)
        Xp = X*t0 + Y*t1 + t2
        Yp = X*t3 + Y*t4 + t5

        return torch.cat((Xp,Yp),3)
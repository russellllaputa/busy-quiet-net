import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

class LoG(nn.Module):

    def __init__(self, channels, sigma, kernel_size):
        super(LoG, self).__init__()

#         kernel_size = 4 * sigma + 1
        self.pad = (kernel_size - 1) // 2
        kernel_size = [kernel_size] * 2
        


        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        mean = (kernel_size[0] - 1) / 2
        
        _x = meshgrids[0] - mean
        _y = meshgrids[1] - mean
        
#         print((meshgrids[0][1][3], meshgrids[1][1][3]))
        
        kernel =  ( -1 / (math.pi * sigma**4) ) * \
                  ( 1- ((_x**2 + _y**2)/(2*sigma**2)) ) *\
                  torch.exp( -(_x**2 + _y**2)/(2*sigma**2) )
#         K = 1.6
#         kernel = ((1/(2*math.pi*sigma**2))*torch.exp(-(_x**2 + _y**2)/(2*sigma**2))) - \
#                  ((1/(2*math.pi*K**2*sigma**2))*torch.exp(-(_x**2 + _y**2)/(2*K**2*sigma**2)))

        # Make sure sum of values in gaussian kernel equals 1.

        kernel = kernel / torch.sum(kernel)
        
#         kernel = kernel.unsqueeze(0)
        

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

#         self.register_buffer('weight', kernel)
        self.weight = nn.Parameter(kernel, requires_grad=True)
        self.groups = channels


    def forward(self, input):

        out = F.conv2d(input, weight=self.weight, groups=self.groups, stride=1)

        return out


class LO(nn.Module):

    def __init__(self, channels, stride):
        super(LO, self).__init__()

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = torch.Tensor([-1,2,-1]) / 3
#         kernel = kernel.unsqueeze(-1)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        
#         self.register_buffer('weight', kernel)
        self.weight = nn.Parameter(kernel, requires_grad=True)
        self.groups = channels
        self.stride = stride


    def forward(self, input):

        return F.conv1d(input, weight=self.weight, groups=self.groups, stride=self.stride)



# class MotionBandPassFilter(nn.Module):
#     def __init__(self, num_segments, channels=3, sigma=1.1, kernel_size=9, First=False):
#         super(MotionBandPassFilter, self).__init__()
        
#         self.First = First
#         self.num_segments = num_segments
#         self.pad = (kernel_size - 1) // 2

#         if First:
#             self.conv_s = LoG(channels=21, sigma=sigma, kernel_size=kernel_size)
#             self.conv_t = LO(channels=3, stride=1)
#         else:
#             self.conv_s = LoG(channels=channels, sigma=sigma, kernel_size=kernel_size)
#             self.conv_t = LO(channels=channels, stride=1)


#     def forward(self, x):
#         if not self.First:
#             # nt, c, h, w
#             t = self.num_segments
#             nt, c, h, w = x.size()
#             n = nt // t
     
#             out = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect') # nt, c, h, w
#             out = self.conv_s(out) # nt, c, h, w
#             out = out = out.view(n, t, c, h*w).transpose(1,3).contiguous().view(-1, c, t) # nhw, c, t
#             out = torch.cat([out[:,:, :1], out, out[:,:,-1:]], 2) # nhw, c, t+2
#             out = self.conv_t(out) # nhw, c, t
#             out = out.view(n, h, w, c, t).permute([0, 4, 3, 1, 2]).contiguous().view(n*t, c, h, w) # nt, c, h, w
#         else:
#             # n, t, d, c, h, w
#             n, t, d, c, h, w = x.size()
#             out = x.view(-1, d*c, h, w) # nt, dc, h, w
#             out = F.pad(out, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
#             out = self.conv_s(out)    # nt, dc, h, w 
#             out = out.view(n*t, d, c, h*w).transpose(1,3).contiguous().view(n*t*h*w, c, d)# nthw, c, d
#             out = self.conv_t(out) # nt, hw, c, d-2
#             out = out.view(n*t, h*w, c, d-2).permute([0,3,2,1]).contiguous().view(n*t, -1, h, w) # nt, (d-2)c, h, w

#         return out
    
# class MotionBandPassFilter(nn.Module):
#     def __init__(self, num_segments, channels=3, sigma=1.1, kernel_size=9, First=False):
#         super(MotionBandPassFilter, self).__init__()
        
#         self.First = First
#         self.num_segments = num_segments
#         self.pad = (kernel_size - 1) // 2

#         if First:
#             self.conv_s = LoG(channels=21, sigma=sigma, kernel_size=kernel_size)
#             self.conv_t = LO(channels=3, stride=1)
#         else:
#             self.conv_s = LoG(channels=channels, sigma=sigma, kernel_size=kernel_size)
#             self.conv_t = LO(channels=channels, stride=1)


#     def forward(self, x):
#         if not self.First:
#             # nt, c, h, w
#             t = self.num_segments
#             nt, c, h, w = x.size()
#             n = nt // t
     
#             out = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect') # nt, c, h, w
#             out = self.conv_s(out) # nt, c, h, w
#             out = out = out.view(n, t, c, h*w).transpose(1,3).contiguous().view(-1, c, t) # nhw, c, t
#             out = torch.cat([out[:,:, :1], out, out[:,:,-1:]], 2) # nhw, c, t+2
#             out = self.conv_t(out) # nhw, c, t
#             out = out.view(n, h, w, c, t).permute([0, 4, 3, 1, 2]).contiguous().view(n*t, c, h, w) # nt, c, h, w
#         else:
#             # n, t, d, c, h, w
#             n, t, d, c, h, w = x.size()
#             out = x.view(-1, d*c, h, w) # nt, dc, h, w
#             out = F.pad(out, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
#             out = self.conv_s(out)    # nt, dc, h, w 
#             out = out.view(n*t, d, c, h*w).transpose(1,2).contiguous().view(n*t, c, d, h*w)# nt, c, d, h*w
#             out = self.conv_t(out) # nt, c, d-2, h*w
#             out = out.transpose(1,2).contiguous().view(n*t, (d-2)*c, h, w) # nt, (d-2)c, h, w

#         return out
    
    
class MotionBandPassFilter(nn.Module):
    def __init__(self, num_segments, channels=3, sigma=1.1, kernel_size=9, three_steps=False):
        super(MotionBandPassFilter, self).__init__()
        
        self.three_steps = three_steps
        self.num_segments = num_segments
        self.pad = (kernel_size - 1) // 2

        if three_steps:
            self.conv_s = LoG(channels=3, sigma=sigma, kernel_size=kernel_size)
            self.conv_t = LO(channels=3, stride=3)
        else:
            self.conv_s = LoG(channels=channels, sigma=sigma, kernel_size=kernel_size)
            self.conv_t = LO(channels=channels, stride=1)


    def forward(self, x):
        if not self.three_steps:
            # nt, c, h, w
            t = self.num_segments
            nt, c, h, w = x.size()
            n = nt // t
     
            out = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect') # nt, c, h, w
            out = self.conv_s(out) # nt, c, h, w
            self.s_out = out
            out = out = out.view(n, t, c, h*w).transpose(1,3).contiguous().view(-1, c, t) # nhw, c, t
            out = torch.cat([out[:,:, :1], out, out[:,:,-1:]], 2) # nhw, c, t+2
            out = self.conv_t(out) # nhw, c, t
            out = out.view(n, h, w, c, t).permute([0, 4, 3, 1, 2]).contiguous().view(n*t, c, h, w) # nt, c, h, w
        else:
            # n, t, 3, c, h, w
            n, t, _3, c, h, w = x.size()
            out = x.view(-1, c, h, w) # nt3, c, h, w
            out = F.pad(out, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
            out = self.conv_s(out)    # nt3, c, h, w
            self.s_out = out
            out = out.view(n, t*3, c, h*w).transpose(1,3).contiguous().view(-1, c, t*3) # nhw, c, t3
            out = self.conv_t(out) # nhw, c, t
            out = out.view(n, h, w, c, t).permute([0, 4, 3, 1, 2]).contiguous().view(n*t, c, h, w) # nt, c, h, w

        return out




#---------------------------------------------------------------------------
# single channel
# import math
# import numbers
# import torch
# from torch import nn
# from torch.nn import functional as F

# class LoG(nn.Module):

#     def __init__(self, channels, sigma, kernel_size):
#         super(LoG, self).__init__()

# #         kernel_size = 4 * sigma + 1
#         self.pad = (kernel_size - 1) // 2
#         kernel_size = [kernel_size] * 2
        


#         kernel = 1
#         meshgrids = torch.meshgrid(
#             [
#                 torch.arange(size, dtype=torch.float32)
#                 for size in kernel_size
#             ]
#         )
#         mean = (kernel_size[0] - 1) / 2
        
#         _x = meshgrids[0] - mean
#         _y = meshgrids[1] - mean
        
# #         print((meshgrids[0][1][3], meshgrids[1][1][3]))
        
#         kernel =  ( -1 / (math.pi * sigma**4) ) * \
#                   ( 1- ((_x**2 + _y**2)/(2*sigma**2)) ) *\
#                   torch.exp( -(_x**2 + _y**2)/(2*sigma**2) )
# #         K = 1.6
# #         kernel = ((1/(2*math.pi*sigma**2))*torch.exp(-(_x**2 + _y**2)/(2*sigma**2))) - \
# #                  ((1/(2*math.pi*K**2*sigma**2))*torch.exp(-(_x**2 + _y**2)/(2*K**2*sigma**2)))

#         # Make sure sum of values in gaussian kernel equals 1.

#         kernel = kernel / torch.sum(kernel)
        
# #         kernel = kernel.unsqueeze(0)
        

#         # Reshape to depthwise convolutional weight
#         kernel = kernel.view(1, 1, *kernel.size())
#         kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

# #         self.register_buffer('weight', kernel)
#         self.weight = nn.Parameter(kernel, requires_grad=True)
#         self.groups = channels


#     def forward(self, input):
#         """
#         Apply gaussian filter to input.
#         Arguments:
#             input (torch.Tensor): Input to apply gaussian filter on.
#         Returns:
#             filtered (torch.Tensor): Filtered output.
#         """

#         out = F.conv2d(input, weight=self.weight, groups=self.groups, stride=1)

    
#         return out


# class LO(nn.Module):

#     def __init__(self, channels, stride):
#         super(LO, self).__init__()

#         # Make sure sum of values in gaussian kernel equals 1.
#         kernel = torch.Tensor([-1,2,-1]) / 3
# #         kernel = kernel.unsqueeze(-1).unsqueeze(-1)

#         # Reshape to depthwise convolutional weight
#         kernel = kernel.view(1, 1, *kernel.size())
#         kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        
# #         self.register_buffer('weight', kernel)
#         self.weight = nn.Parameter(kernel, requires_grad=True)
#         self.groups = channels
#         self.stride = stride


#     def forward(self, input):
#         """
#         Apply gaussian filter to input.
#         Arguments:
#             input (torch.Tensor): Input to apply gaussian filter on.
#         Returns:
#             filtered (torch.Tensor): Filtered output.
#         """
#         return F.conv1d(input, weight=self.weight, groups=self.groups, stride=self.stride)



# class MotionBandPassFilter(nn.Module):
#     def __init__(self, num_segments, channels=3, sigma=0.7, kernel_size=9, three_steps=False):
#         super(MotionBandPassFilter, self).__init__()
        
#         self.three_steps = three_steps
#         self.num_segments = num_segments
#         self.pad = (kernel_size - 1) // 2

        
#         self.conv_s = LoG(channels=1, sigma=sigma, kernel_size=kernel_size)
#         if three_steps:
#             self.conv_t = LO(channels=1, stride=3)
#         else:
#             self.conv_t = LO(channels=1, stride=1)


#     def forward(self, x):
#         if not self.three_steps:
#             # nt, c, h, w
#             t = self.num_segments
#             nt, c, h, w = x.size()
#             n = nt // t
     
#             out = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect') # nt, c, h, w
#             out = out.view(nt*c, 1, h+2*self.pad, w+2*self.pad) # ntc, 1, h, w
#             out = self.conv_s(out) # ntc, 1, h, w
#             out = out = out.view(n, t, c, h*w).transpose(1,3).contiguous().view(-1, 1, t) # nhwc, 1, t
#             out = torch.cat([out[:,:, :1], out, out[:,:,-1:]], 2)
#             out = self.conv_t(out).view(n,c,t,h,w) # nhwc, 1, t
#             out = out.view(n, h, w, c, t).permute([0, 4, 3, 1, 2]).contiguous().view(n*t, c, h, w) # nt, c, h, w
#         else:
#             # n, t, 3, c, h, w
#             n, t, _3, c, h, w = x.size()
#             out = x.view(-1, 1, h, w) # nt3c, 1, h, w
#             out = F.pad(out, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
#             out = self.conv_s(out) # nt3c, 1, h, w
#             out = out.view(n, t*3, c, h*w).transpose(1,3).contiguous().view(-1, 1, t*3) # nhwc, 1, t3
#             out = self.conv_t(out) # nhwc, 1, t
#             out = out.view(n, h, w, c, t).permute([0, 4, 3, 1, 2]).contiguous().view(n*t, c, h, w) # nt, c, h, w

#         return out
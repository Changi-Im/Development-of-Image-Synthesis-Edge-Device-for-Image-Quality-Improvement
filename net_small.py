import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import fusion_strategy
from binary_fractions import Binary

# Depthwise convolution
class Depthwise_conv(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(Depthwise_conv, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, groups=in_channels, bias = False, padding = (1,1))

    def forward(self, x):
        out = self.depthwise(x)

        return out
    
# Pointwise convolution
class Pointwise_conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size = 1, is_last=False):
        super(Pointwise_conv, self).__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias = False)
        self.is_last = is_last

    def forward(self, x):
        out = self.pointwise(x)
        if self.is_last is False:
            out = F.relu(out, inplace=True)

        return out

# Depthwise seperable convolution unit
class Separable_conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(Separable_conv, self).__init__()
        self.depthwise = Depthwise_conv(in_channels, kernel_size, stride)
        self.pointwise = Pointwise_conv(in_channels = in_channels, out_channels = out_channels, stride = stride, is_last = is_last)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
    
        return out
    
# Dense depthwise seperable convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = Separable_conv(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out
    
# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 3
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out
    
# DenseFuse network
class DenseFuse_net(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(DenseFuse_net, self).__init__()
        denseblock = DenseBlock
        #nb_filter = [16, 64, 32, 16]
        nb_filter = [3, 12, 6, 3]
        kernel_size = 3
        stride = 1

        # encoder1
        self.conv1 = Separable_conv(input_nc, nb_filter[0], kernel_size, stride)
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)
        
        # encoder2
        self.conv1_1 = Separable_conv(input_nc, nb_filter[0], kernel_size, stride)
        self.DB2 = denseblock(nb_filter[0], kernel_size, stride)
        
        # decoder
        self.conv2 = Separable_conv(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.conv3 = Separable_conv(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = Separable_conv(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = Separable_conv(nb_filter[3], output_nc, kernel_size, stride)

    def encoder1(self, input):
        x1 = self.conv1(input)
        x_DB1 = self.DB1(x1)
        return [x_DB1]
    
    def encoder2(self, input):
        x2 = self.conv1_1(input)
        x_DB2 = self.DB2(x2)
        return [x_DB2]

    # def fusion(self, en1, en2, strategy_type='addition'):
    #     # addition
    #     if strategy_type is 'attention_weight':
    #         # attention weight
    #         fusion_function = fusion_strategy.attention_fusion_weight
    #     else:
    #         fusion_function = fusion_strategy.addition_fusion
    #
    #     f_0 = fusion_function(en1[0], en2[0])
    #     return [f_0]

    def fusion(self, en1, en2, strategy_type='addition'):
        f_0 = en1[0] + en2[0]
        return [f_0]

    def decoder(self, f_en):
        x2 = self.conv2(f_en[0])
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        output = self.conv5(x4)

        return [output]


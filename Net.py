import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU
import fusion_strategy

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_relu=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))  # 下取整
        self.reflection_pad = nn.ReflectionPad2d(
            [reflection_padding, reflection_padding, reflection_padding, reflection_padding])
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.dropout = nn.Dropout2d(p=0.5)
        self.use_relu = use_relu

    def forward(self, x):
        # print(x.shape)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.use_relu:
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out



# Dense Block unit
class res2net_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_size_neck, stride, width):
        super(res2net_Block, self).__init__()
        self.width = width
        convs = []
        self.conv1 = ConvLayer(in_channels, self.width * 4, kernel_size, stride, use_relu=True)
        for i in range(3):
            convs.append(ConvLayer(self.width, self.width, kernel_size_neck, stride, use_relu=True))
        self.convs = nn.ModuleList(convs)
        self.conv2 = ConvLayer(self.width * 4, out_channels, kernel_size, stride, use_relu=False)
        self.conv3 = ConvLayer(in_channels, out_channels, kernel_size, stride, use_relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out1 = self.conv1(x)
        spx = torch.split(out1, self.width, 1)
        for i in range(3):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out2 = torch.cat((out, spx[3]), 1)
        out3 = self.conv2(out2)
        out4 = self.conv3(residual)
        OUT = self.relu(out3 + out4)
        return OUT


# DenseFuse network
class Rse2Net_atten_fuse(nn.Module):
    def __init__(self,output_nc=1):
        super(Rse2Net_atten_fuse, self).__init__()
        resblock = res2net_Block
        width = [2, 4, 8, 16]
        encoder_inchannel = [1, 16, 48]
        encoder_outchannel = [16, 32, 64]
        kernel_size_1 = 1
        kernel_size_2 = 3
        decoder_channel = [112, 64, 32, 16]
        stride = 1

        # encoder
        self.encoder_pad = nn.ReflectionPad2d([1, 1, 1, 1])
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=encoder_inchannel[0], out_channels=encoder_outchannel[0], kernel_size=3),
            ReLU(inplace=True))
        self.RB1 = resblock(encoder_inchannel[1], encoder_outchannel[1], kernel_size_1, kernel_size_2, stride, width[1])
        self.RB2 = resblock(encoder_inchannel[2], encoder_outchannel[2], kernel_size_1, kernel_size_2, stride, width[2])


        # decoder
        self.conv1 = ConvLayer(decoder_channel[0], decoder_channel[1], kernel_size_2, stride,use_relu=True)
        self.conv2 = ConvLayer(decoder_channel[1], decoder_channel[2], kernel_size_2, stride,use_relu=True)
        self.conv3 = ConvLayer(decoder_channel[2], decoder_channel[3], kernel_size_2, stride,use_relu=True)
        self.conv4 = ConvLayer(decoder_channel[3], output_nc, kernel_size_2, stride,use_relu=True)


    def encoder(self, input):
        input_pad = self.encoder_pad(input)
        x0 = self.encoder_conv(input_pad)
        x1 = self.RB1(x0)
        x01 = torch.cat([x0, x1], dim=1)
        x2 = self.RB2(x01)
        x012 = torch.cat([x01, x2], dim=1)

        return x012



    def fusion_atten(self, en1, en2 ,strategy_type='add'):
        if strategy_type is 'atten':
            # attention weight
            fusion_function = fusion_strategy.attention_fusion_weight
            f_0 = fusion_function(en1, en2)
        else:
            # addition
            fusion_function = fusion_strategy.addition_fusion
            f_0 = fusion_function(en1, en2)

        return f_0


    def decoder(self, f_en_atten):
        x2 = self.conv1(f_en_atten)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        output = self.conv4(x4)
        return output

from torch import nn
from collections import OrderedDict


def conv1x1(in_channels, out_channels, stride=1, bias=True):

    return nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):

    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


class ConvInNormLeReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(ConvInNormLeReLU, self).__init__(
            OrderedDict(
                [
                    ('Conv', conv3x3(in_channels, out_channels)),
                    ('InNorm', nn.InstanceNorm3d(out_channels)),
                    ('LeReLU', nn.LeakyReLU(inplace=True))
                ]
            )
        )


class LevelBlock(nn.Sequential):

    def __init__(self, in_channels, mid_channels, out_channels, dilation=(1, 1)):
        super(LevelBlock, self).__init__(
            OrderedDict(
                [
                    ('ConvBnRelu1', ConvInNormLeReLU(in_channels, mid_channels, dilation=dilation[0])),
                    ('ConvBnRelu2', ConvInNormLeReLU(mid_channels, out_channels, dilation=dilation[1]))
                ])
        )

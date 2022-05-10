from torch import nn
from collections import OrderedDict


def conv1x1(in_channels, out_channels):
    """ 3D convolution which uses a kernel size of 1"""

    return nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1))


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1, bias=False):
    """ 3D convolution which uses a kernel size of 3"""

    return nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv_single(in_channels, out_channels=1, stride=1, groups=1, dilation=1, bias=False):
    """ 3D convolution which uses a kernel size of 3"""

    return nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


class ConvInNormLeReLU(nn.Sequential):
    """
    This class stacks a 3D Convolution, Instance Normalization and Leaky ReLU layers.

    Params
    ******
        - in_channels: Number of input channels
        - out_channels: Number of output channels

    """

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
    """
    This class stacks two blocks of ConvInNormLeReLU (3D Convolution, Instance Normalization and Leaky ReLU layers).

    Params
    ******
        - in_channels: Number of input channels
        - mid_channels: Number of channels between the first and the second block
        - out_channels: Number of output channels

    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(LevelBlock, self).__init__(
            OrderedDict(
                [
                    ('ConvInNormLRelu1', ConvInNormLeReLU(in_channels, mid_channels)),
                    ('ConvInNormLRelu2', ConvInNormLeReLU(mid_channels, out_channels))
                ])
        )


class ConvBatchNormReLU(nn.Sequential):
    """
    This class stacks a 3D Convolution, Batch Normalization and ReLU layers.

    Params
    ******
        - in_channels: Number of input channels
        - out_channels: Number of output channels

    """
    def __init__(self, in_channels, out_channels):
        super(ConvBatchNormReLU, self).__init__(
            OrderedDict(
                [
                    ('Conv', conv3x3(in_channels, out_channels)),
                    ('BatchNorm', nn.BatchNorm3d(out_channels)),
                    ('ReLU', nn.ReLU())
                ]
            )
        )


class UBlock(nn.Sequential):
    """
    This class stacks two blocks of ConvBatchNormReLU (3D Convolution, Batch Normalization and ReLU layers).

    Params
    ******
        - in_channels: Number of input channels
        - mid_channels: Number of channels between the first and the second block
        - out_channels: Number of output channels

    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(UBlock, self).__init__(
            OrderedDict(
                [
                    ('ConvBnRelu1', ConvBatchNormReLU(in_channels, mid_channels)),
                    ('ConvBnRelu2', ConvBatchNormReLU(mid_channels, out_channels))
                ])
        )


class AttentionGate(nn.Sequential):
    """
    This class stacks a 3D Convolution, Batch Normalization and ReLU layers.

    Params
    ******
        - in_channels: Number of input channels
        - out_channels: Number of output channels

    """
    def __init__(self, in_channels):
        super(AttentionGate, self).__init__(
            OrderedDict(
                [
                    ('ReLU', nn.ReLU()),
                    ('Conv', conv_single(in_channels)),
                    ('InNorm', nn.InstanceNorm3d(in_channels)),
                    ('Sigmoid', nn.Sigmoid())
                ]
            )
        )


class LevelBlock2x2(nn.Sequential):
    """
    This class stacks two blocks of ConvInNormLeReLU (3D Convolution, Instance Normalization and Leaky ReLU layers).

    Params
    ******
        - in_channels: Number of input channels
        - mid_channels: Number of channels between the first and the second block
        - out_channels: Number of output channels

    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(LevelBlock2x2, self).__init__(
            OrderedDict(
                [
                    ('Conv1', nn.Conv3d(in_channels, mid_channels, kernel_size=(2, 2, 2), stride=1, padding=1)),
                    ('InNorm1', nn.InstanceNorm3d(mid_channels)),
                    ('LeakyReLU1', nn.LeakyReLU()),
                    ('Conv2', nn.Conv3d(mid_channels, out_channels, kernel_size=(2, 2, 2), stride=1, padding=0)),
                    ('InNorm2', nn.InstanceNorm3d(out_channels)),
                    ('LeakyReLU2', nn.LeakyReLU())
                ])
        )


class FullyConnectedClassifier(nn.Sequential):
    """
    This class stacks a 3D Convolution, Instance Normalization and Leaky ReLU layers.

    Params
    ******
        - classes: Number of classes

    """

    def __init__(self, width, middle_num_neurons=128, classes=2):
        super(FullyConnectedClassifier, self).__init__(
            OrderedDict(
                [
                    ('layer1', nn.Linear(in_features=16*width*10*14*10, out_features=middle_num_neurons)),
                    ('dropout', nn.Dropout(.2)),
                    ('ReLU1', nn.ReLU()),
                    ('layer2', nn.Linear(in_features=middle_num_neurons, out_features=64)),
                    ('ReLU2', nn.ReLU()),
                    ('layer3', nn.Linear(in_features=64, out_features=1))
                ]
            )
        )

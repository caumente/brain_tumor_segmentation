import torch.nn as nn
import torch

"""
Implementation of this model is borrowed and modified (to support multi-channels and latest pytorch version) from:
https://github.com/Dawn90/V-Net.pytorch
"""


def passthrough(x):
    """
    This layer is useful to not do anything action over the input
    """
    return x

def make_convs_residuals(n_channels, n_convolutions, elu):
    layers = []
    for _ in range(n_convolutions):
        layers.append(LUConv(n_channels, elu))
    return nn.Sequential(*layers)

def ActivationLayer(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, n_channels, elu):
        super(LUConv, self).__init__()
        self.conv = nn.Conv3d(n_channels, n_channels, kernel_size=5, padding=2)
        self.norm = torch.nn.BatchNorm3d(n_channels)
        self.activation = ActivationLayer(elu, n_channels)

    def forward(self, x):
        out = self.activation(self.norm(self.conv(x)))
        return out


class InputTransition(nn.Module):
    def __init__(self, in_channels, elu):
        super(InputTransition, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.num_features, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(num_features=self.num_features)
        self.relu1 = ActivationLayer(elu, self.num_features)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)

        repeat_rate = int(self.num_features / self.in_channels)
        x_residual = x.repeat(1, repeat_rate, 1, 1, 1)

        return self.relu1(torch.add(out, x_residual))


class DownTransition(nn.Module):
    def __init__(self, in_channels, n_convolutions, elu, dropout=False):
        super(DownTransition, self).__init__()
        out_channels = 2 * in_channels

        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = torch.nn.BatchNorm3d(out_channels)
        self.activation = ActivationLayer(elu, out_channels)

        if dropout:
            self.do1 = nn.Dropout3d()
        else:
            self.do1 = passthrough

        self.ops = make_convs_residuals(out_channels, n_convolutions, elu)

    def forward(self, x):
        down = self.activation(self.norm(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.activation(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)

        self.bn1 = torch.nn.BatchNorm3d(outChans // 2)
        self.do2 = nn.Dropout3d()
        self.activation1 = ActivationLayer(elu, outChans // 2)
        self.activation2 = ActivationLayer(elu, outChans)

        if dropout:
            self.do1 = nn.Dropout3d()
        else:
            self.do1 = passthrough

        self.ops = make_convs_residuals(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.activation1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.activation2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, in_channels, classes, elu):
        super(OutputTransition, self).__init__()
        self.classes = classes
        self.conv1 = nn.Conv3d(in_channels, classes, kernel_size=5, padding=2)
        self.norm = torch.nn.BatchNorm3d(classes)

        self.conv2 = nn.Conv3d(classes, classes, kernel_size=1)
        self.activation = ActivationLayer(elu, classes)

    def forward(self, x):
        # convolve 32 down to channels as the desired classes
        out = self.activation(self.norm(self.conv1(x)))
        out = self.conv2(out)
        return out


class VNet(nn.Module):
    """
    Implementations based on the Vnet paper: https://arxiv.org/abs/1606.04797
    """
    name = 'Vnet'

    def __init__(self, elu=True, sequences=4, regions=3):
        super(VNet, self).__init__()

        self.in_tr = InputTransition(sequences, elu=elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, regions, elu)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)

        return out




if __name__ == "__main__":
    seq_input = torch.rand(1, 4, 160, 224, 160)
    seq_ouput = torch.rand(1, 3, 160, 224, 160)

    model = VNet(elu=True, sequences=4, regions=3)
    preds = model(seq_input)
    from flopth import flopth
    flops, params = flopth(model, in_size=((4, 160, 224, 160),), show_detail=False, bare_number=True)
    print(flops/1000000000000)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))
    # from fvcore.nn import FlopCountAnalysis
    #
    # flops = FlopCountAnalysis(model, seq_input)
    # print(flops.total())
    #
    # assert seq_ouput.shape == preds.shape
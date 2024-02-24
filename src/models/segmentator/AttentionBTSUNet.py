import torch
from torch import nn

from src.models.layers import ConvInNormLeReLU, AttentionGate
from src.models.layers import LevelBlock
from src.models.layers import conv1x1, conv3x3


class AttentionBTSUNet(nn.Module):
    """
    This class implements a variation of BTS UNet network. Main modifications are the inclusions of AttentionGates
    """

    name = "Attention BTS U-Net"

    def __init__(self, sequences, regions, width):
        super(AttentionBTSUNet, self).__init__()

        widths = [width * 2 ** i for i in range(4)]

        # Encoders
        self.encoder1 = LevelBlock(sequences, widths[0] // 2, widths[0])
        self.encoder2 = LevelBlock(widths[0], widths[1] // 2, widths[1])
        self.encoder3 = LevelBlock(widths[1], widths[2] // 2, widths[2])
        self.encoder4 = LevelBlock(widths[2], widths[3] // 2, widths[3])

        # Bottleneck
        self.bottleneck = LevelBlock(widths[3], widths[3], widths[3])
        self.bottleneck2 = ConvInNormLeReLU(widths[3] * 2, widths[2])

        # Decoders
        self.decoder3 = LevelBlock(widths[2] * 2, widths[2], widths[1])
        self.decoder2 = LevelBlock(widths[1] * 2, widths[1], widths[0])
        self.decoder1 = LevelBlock(widths[0] * 2, widths[0], widths[0] // 2)

        # Attention gates
        self.strided_conv3 = conv3x3(widths[2], widths[2], stride=2)
        self.strided_conv2 = conv3x3(widths[1], widths[1], stride=2)
        self.strided_conv1 = conv3x3(widths[0], widths[0], stride=2)
        self.att3 = AttentionGate(in_channels=widths[2])
        self.att2 = AttentionGate(in_channels=widths[1])
        self.att1 = AttentionGate(in_channels=widths[0])

        # Upsample, downsample and output steps
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.downsample = nn.MaxPool3d(2, 2)
        self.output = conv1x1(widths[0] // 2, regions)

        self.weights_initialization()

    def weights_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoding phase
        e1 = self.encoder1(x)
        p1 = self.downsample(e1)
        e2 = self.encoder2(p1)
        p2 = self.downsample(e2)
        e3 = self.encoder3(p2)
        p3 = self.downsample(e3)
        e4 = self.encoder4(p3)

        # Bottleneck phase
        bottleneck = self.bottleneck(e4)
        bottleneck2 = self.bottleneck2(torch.cat([e4, bottleneck], dim=1))

        # attention 3
        alpha3 = torch.add(self.strided_conv3(e3), bottleneck2)
        attention_grid3 = self.upsample(self.att3(alpha3))
        e3_attentioned = torch.mul(e3, attention_grid3)

        # Decoding phase + skip connections 3
        up3 = self.upsample(bottleneck2)
        d3 = self.decoder3(torch.cat([e3_attentioned, up3], dim=1))

        # attention 2
        alpha2 = torch.add(self.strided_conv2(e2), d3)
        attention_grid2 = self.upsample(self.att2(alpha2))
        e2_attentioned = torch.mul(e2, attention_grid2)

        up2 = self.upsample(d3)
        d2 = self.decoder2(torch.cat([e2_attentioned, up2], dim=1))

        # attention 1
        alpha1 = torch.add(self.strided_conv1(e1), d2)
        attention_grid1 = self.upsample(self.att1(alpha1))
        e1_attentioned = torch.mul(e1, attention_grid1)

        up1 = self.upsample(d2)
        d1 = self.decoder1(torch.cat([e1_attentioned, up1], dim=1))

        # Output
        output = self.output(d1)

        return output


if __name__ == "__main__":
    # Defining variables
    seq_input = torch.rand(1, 4, 160, 224, 160)
    seq_ouput = torch.rand(1, 3, 160, 224, 160)
    model = AttentionBTSUNet(sequences=4, regions=3, width=24)
    preds = model(seq_input)

    # Getting TFLOPs
    from flopth import flopth
    flops, params = flopth(model, in_size=((4, 160, 224, 160),), show_detail=False, bare_number=True)
    print('Number of TFLOPs: {:.3f}'.format(flops / 1e12))

    # Getting number of trainable parameters
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('Model size: {:.3f}MB'.format(size_all_mb))

    # Validating output dimensions
    print(f"Input shape: {seq_input.shape}")
    if isinstance(preds, list):
        for n, p in enumerate(preds):
            print(f"Output shape (n): {p.shape}")
            assert seq_ouput.shape == p.shape
    else:
        print(f"Output shape: {preds.shape}")
        assert seq_ouput.shape == preds.shape

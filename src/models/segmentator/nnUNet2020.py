import torch
from torch import nn

from src.models.layers import LevelBlock
from src.models.layers import conv1x1


class nnUNet2020(nn.Module):
    """
    This class implements nnU-Net network. The winner network of the BraTS 2020 challenge. More details can be found in
    the following paper https://arxiv.org/abs/2011.00848
    """

    name = "nn-UNet2020"

    def __init__(self, sequences, regions):
        super(nnUNet2020, self).__init__()

        widths = [32, 64, 128, 256, 320]

        # Encoders
        self.encoder1 = LevelBlock(sequences, widths[0], widths[0])
        self.encoder2 = LevelBlock(widths[0], widths[1], widths[1])
        self.encoder3 = LevelBlock(widths[1], widths[2], widths[2])
        self.encoder4 = LevelBlock(widths[2], widths[3], widths[3])
        self.encoder5 = LevelBlock(widths[3], widths[4], widths[4])

        # Bottleneck
        self.bottleneck = LevelBlock(widths[4], widths[4], widths[4])

        # Decoders
        self.decoder5 = LevelBlock(widths[4] + widths[4], widths[3], widths[3])
        self.decoder4 = LevelBlock(widths[3] + widths[3], widths[2], widths[2])
        self.decoder3 = LevelBlock(widths[2] + widths[2], widths[1], widths[1])
        self.decoder2 = LevelBlock(widths[1] + widths[1], widths[0], widths[0])
        self.decoder1 = LevelBlock(widths[0] + widths[0], widths[0], widths[0])

        # Upsamplers
        self.upsample5 = nn.ConvTranspose3d(widths[4], widths[4], kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose3d(widths[3], widths[3], kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose3d(widths[2], widths[2], kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose3d(widths[1], widths[1], kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose3d(widths[0], widths[0], kernel_size=2, stride=2)

        # Downsample and output steps
        self.downsample = nn.MaxPool3d(2, 2)

        # Outputs
        self.output4 = nn.Sequential(
            nn.ConvTranspose3d(widths[2], widths[2], kernel_size=8, stride=8),
            conv1x1(widths[2], regions)
        )
        self.output3 = nn.Sequential(
            nn.ConvTranspose3d(widths[1], widths[1], kernel_size=4, stride=4),
            conv1x1(widths[1], regions)
        )
        self.output2 = nn.Sequential(
            nn.ConvTranspose3d(widths[0], widths[0], kernel_size=2, stride=2),
            conv1x1(widths[0], regions)
        )
        self.output1 = conv1x1(widths[0], regions)

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
        p4 = self.downsample(e4)
        e5 = self.encoder5(p4)
        p5 = self.downsample(e5)

        # Bottleneck
        bottleneck = self.bottleneck(p5)

        # Decoding phase + skip connections
        up5 = self.upsample5(bottleneck)
        d5 = self.decoder5(torch.cat([e5, up5], dim=1))
        up4 = self.upsample4(d5)
        d4 = self.decoder4(torch.cat([e4, up4], dim=1))
        up3 = self.upsample3(d4)
        d3 = self.decoder3(torch.cat([e3, up3], dim=1))
        up2 = self.upsample2(d3)
        d2 = self.decoder2(torch.cat([e2, up2], dim=1))
        up1 = self.upsample1(d2)
        d1 = self.decoder1(torch.cat([e1, up1], dim=1))

        # Output
        output4 = self.output4(d4)
        output3 = self.output3(d3)
        output2 = self.output2(d2)
        output1 = self.output1(d1)

        return [output4, output3, output2, output1]


if __name__ == "__main__":
    # Defining variables
    seq_input = torch.rand(1, 4, 160, 224, 160)
    seq_ouput = torch.rand(1, 3, 160, 224, 160)
    model = nnUNet2020(sequences=4, regions=3)
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

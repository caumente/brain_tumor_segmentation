import torch
from torch import nn
from src.models.layers import conv1x1
from src.models.layers import LevelBlock
from src.models.layers import ConvInNormLeReLU


class ShallowUNet(nn.Module):
    """
    This class implements a variation of 3D Unet network. Main modifications are:
        - Replacement of ReLU activation layer by LeakyReLU
        - Use of instance normalization to ensure a normalization by each sequence

    """

    name = "Shallow U-Net"

    def __init__(self, sequences, regions, width, deep_supervision):
        super(ShallowUNet, self).__init__()

        self.deep_supervision = deep_supervision
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

        # Upsample, downsample and output steps
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.downsample = nn.MaxPool3d(2, 2)


        # Output
        if self.deep_supervision:
            self.output3 = nn.Sequential(
                nn.ConvTranspose3d(widths[1], widths[1], kernel_size=4, stride=4),
                conv1x1(widths[1], regions)
            )
            self.output2 = nn.Sequential(
                nn.ConvTranspose3d(widths[0], widths[0], kernel_size=2, stride=2),
                conv1x1(widths[0], regions)
            )
        self.output1 = conv1x1(widths[0] // 2, regions)

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

        # Decoding phase + skip connections
        up3 = self.upsample(bottleneck2)
        d3 = self.decoder3(torch.cat([e3, up3], dim=1))
        up2 = self.upsample(d3)
        d2 = self.decoder2(torch.cat([e2, up2], dim=1))
        up1 = self.upsample(d2)
        d1 = self.decoder1(torch.cat([e1, up1], dim=1))

        # Output
        if self.deep_supervision:
            output3 = self.output3(d3)
            output2 = self.output2(d2)
            output1 = self.output1(d1)

            return [output3, output2, output1]
        else:
            output1 = self.output1(d1)

            return output1



def test():
    seq_input = torch.rand(1, 4, 160, 224, 160)
    seq_ouput = torch.rand(1, 3, 160, 224, 160)

    model = ShallowUNet(sequences=4, regions=3, width=24, deep_supervision=False)
    preds = model(seq_input)

    print(seq_input.shape)
    if model.deep_supervision:
        for p in preds:
            print(p.shape)
            assert seq_ouput.shape == p.shape
    else:
        print(preds.shape)
        assert seq_ouput.shape == preds.shape


if __name__ == "__main__":
    test()

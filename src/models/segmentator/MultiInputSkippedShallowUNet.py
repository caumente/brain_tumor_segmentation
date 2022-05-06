import torch
from torch import nn
from src.models.layers import conv1x1
from src.models.layers import LevelBlock
from src.models.layers import ConvInNormLeReLU


class MultiInputSkippedShallowUNet(nn.Module):
    """
    This class implements a variation of 3D Unet network. Main modifications are:
        - Replacement of ReLU activation layer by LeakyReLU
        - Use of instance normalization to ensure a normalization by each sequence

    """

    name = "Multi-Input Skipped Shallow U-Net"

    def __init__(self, sequences, regions, width, deep_supervision):
        super(MultiInputSkippedShallowUNet, self).__init__()

        self.deep_supervision = deep_supervision
        widths = [width * 2 ** i for i in range(4)]

        # Encoders
        self.encoder1_simple = LevelBlock(sequences//2, widths[0] // 2, widths[0])
        self.encoder1_multiple = LevelBlock(sequences, widths[0] // 2, widths[0])
        self.encoder2 = LevelBlock(widths[0], widths[1] // 2, widths[1])
        self.encoder3 = LevelBlock(widths[1], widths[2] // 2, widths[2])
        self.encoder4 = LevelBlock(widths[2], widths[3] // 2, widths[3])

        # Bottleneck
        self.bottleneck = LevelBlock(widths[3], widths[3], widths[3])
        self.bottleneck2 = ConvInNormLeReLU(widths[3] * 4, widths[2])

        # Decoders
        self.decoder3 = LevelBlock(widths[2] * 4, widths[2], widths[1])
        self.decoder2 = LevelBlock(widths[1] * 4, widths[1], widths[0])
        self.decoder1 = LevelBlock(widths[0] * 4, widths[0], widths[0] // 2)

        # Upsample, downsample and output steps
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.downsample = nn.MaxPool3d(2, 2)


        # Output
        self.output3 = nn.Sequential(
            nn.ConvTranspose3d(widths[1], widths[1], kernel_size=4, stride=4),
            LevelBlock(widths[1], regions, regions)
        )
        self.output2 = nn.Sequential(
            nn.ConvTranspose3d(widths[0], widths[0], kernel_size=2, stride=2),
            LevelBlock(widths[0], regions, regions)
        )
        self.output1 = LevelBlock(widths[0] // 2, regions, regions)

        self.squeeze_multi_output = conv1x1(3 * regions, regions)
        self.weights_initialization()

    def weights_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Multi-3D UNet
        """

        """
        Encodings
        """

        # Encoding phase U-Net 1
        e11 = self.encoder1_simple(x[:, :2, :, :, :])
        p11 = self.downsample(e11)
        e21 = self.encoder2(p11)
        p21 = self.downsample(e21)
        e31 = self.encoder3(p21)
        p31 = self.downsample(e31)
        e41 = self.encoder4(p31)

        bottleneck1 = self.bottleneck(e41)

        # Encoding phase U-Net 2
        e12 = self.encoder1_simple(x[:, 2:, :, :, :])
        p12 = self.downsample(e12)
        e22 = self.encoder2(p12)
        p22 = self.downsample(e22)
        e32 = self.encoder3(p22)
        p32 = self.downsample(e32)
        e42 = self.encoder4(p32)

        bottleneck2 = self.bottleneck(e42)

        # Encoding phase U-Net 3
        e13 = self.encoder1_multiple(x)
        p13 = self.downsample(e13)
        e23 = self.encoder2(p13)
        p23 = self.downsample(e23)
        e33 = self.encoder3(p23)
        p33 = self.downsample(e33)
        e43 = self.encoder4(p33)

        bottleneck3 = self.bottleneck(e43)


        """
        Decodings
        """

        # Decoding phase + skip connections
        bottleneck21 = self.bottleneck2(torch.cat([e41, e42, e43, bottleneck1], dim=1))
        up31 = self.upsample(bottleneck21)
        d31 = self.decoder3(torch.cat([e31, e32, e33, up31], dim=1))
        up21 = self.upsample(d31)
        d21 = self.decoder2(torch.cat([e21, e22, e23, up21], dim=1))
        up11 = self.upsample(d21)
        d11 = self.decoder1(torch.cat([e11, e12, e13, up11], dim=1))

        if self.deep_supervision:
            output31 = self.output3(d31)
            output21 = self.output2(d21)
            output11 = self.output1(d11)

            output1 = [output31, output21, output11]
        else:
            output1 = self.output1(d11)


        # Decoding phase + skip connections
        bottleneck22 = self.bottleneck2(torch.cat([e41, e42, e43, bottleneck2], dim=1))
        up32 = self.upsample(bottleneck22)
        d32 = self.decoder3(torch.cat([e31, e32, e33, up32], dim=1))
        up22 = self.upsample(d32)
        d22 = self.decoder2(torch.cat([e21, e22, e23, up22], dim=1))
        up12 = self.upsample(d22)
        d12 = self.decoder1(torch.cat([e11, e12, e13, up12], dim=1))

        if self.deep_supervision:
            output32 = self.output3(d32)
            output22 = self.output2(d22)
            output12 = self.output1(d12)

            output2 = [output32, output22, output12]
        else:
            output2 = self.output1(d12)


        # Decoding phase + skip connections
        bottleneck23 = self.bottleneck2(torch.cat([e41, e42, e43, bottleneck3], dim=1))
        up33 = self.upsample(bottleneck23)
        d33 = self.decoder3(torch.cat([e31, e32, e33, up33], dim=1))
        up23 = self.upsample(d33)
        d23 = self.decoder2(torch.cat([e21, e22, e23, up23], dim=1))
        up13 = self.upsample(d23)
        d13 = self.decoder1(torch.cat([e11, e12, e13, up13], dim=1))

        if self.deep_supervision:
            output33 = self.output3(d33)
            output23 = self.output2(d23)
            output13 = self.output1(d13)

            output3 = [output33, output23, output13]
        else:
            output3 = self.output1(d13)


        """
        Results concatenation
        """
        if not self.deep_supervision:
            concatenated_output = torch.cat([output1, output2, output3], dim=1)
            concatenated_output = self.squeeze_multi_output(concatenated_output)

        else:
            concatenated_output = [self.squeeze_multi_output(torch.cat([i, j, k], dim=1)) for i, j, k in zip(output1, output2, output3)]

        return concatenated_output


def test():
    seq_input = torch.rand(1, 4, 160, 224, 160)
    seq_ouput = torch.rand(1, 3, 160, 224, 160)

    model = MultiInputSkippedShallowUNet(sequences=4, regions=3, width=6, deep_supervision=False)
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
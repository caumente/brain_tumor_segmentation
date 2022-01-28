import torch
from torch import nn
from src.models.layers import conv1x1
from src.models.layers import LevelBlock
from src.models.layers import ConvInNormLeReLU


class DeepUNet(nn.Module):
    """
    This class implements a variation of 3D Unet network. Main modifications are:
        - Six levels of depth instead of four
        - Replacement of ReLU activation layer by LeakyReLU
        - Adaptive pooling layers to be able to upsample the images
        - Use of instance normalization to ensure a normalization by each sequence

    It is prepared to get as input images which resolution in 160, 224, 160.
    """

    name = "Deep U-Net"

    def __init__(self, sequences, regions, width):
        super(DeepUNet, self).__init__()

        widths = [width * 2 ** i for i in range(6)]
        widths[-1] = widths[-2]

        # Encoders
        self.encoder1 = LevelBlock(sequences, widths[0] // 2, widths[0])
        self.encoder2 = LevelBlock(widths[0], widths[1] // 2, widths[1])
        self.encoder3 = LevelBlock(widths[1], widths[2] // 2, widths[2])
        self.encoder4 = LevelBlock(widths[2], widths[3] // 2, widths[3])
        self.encoder5 = LevelBlock(widths[3], widths[4] // 2, widths[4])
        self.encoder6 = LevelBlock(widths[4], widths[5] // 2, widths[5])

        # Bottleneck
        self.bottleneck = LevelBlock(widths[5], widths[5], widths[5])
        self.bottleneck2 = ConvInNormLeReLU(widths[5] * 2, widths[4])

        # Decoders
        self.decoder5 = LevelBlock(widths[4] * 2, widths[4], widths[3])
        self.decoder4 = LevelBlock(widths[3] * 2, widths[3], widths[2])
        self.decoder3 = LevelBlock(widths[2] * 2, widths[2], widths[1])
        self.decoder2 = LevelBlock(widths[1] * 2, widths[1], widths[0])
        self.decoder1 = LevelBlock(widths[0] * 2, widths[0], widths[0] // 2)

        # Downsamplings
        self.downsample = nn.MaxPool3d(2, 2)
        self.ads_96 = nn.AdaptiveMaxPool3d(output_size=(80, 112, 80))
        self.ads_64 = nn.AdaptiveMaxPool3d(output_size=(40, 56, 40))
        self.ads_48 = nn.AdaptiveMaxPool3d(output_size=(20, 28, 20))
        self.ads_32 = nn.AdaptiveMaxPool3d(output_size=(20, 28, 20))
        self.ads_24 = nn.AdaptiveMaxPool3d(output_size=(10, 14, 10))
        self.ads_16 = nn.AdaptiveMaxPool3d(output_size=(10, 14, 10))

        # Upsamplings
        self.upsample_2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.upsample_128 = nn.Upsample(size=(160, 224, 160), mode="trilinear", align_corners=True)
        self.upsample_96 = nn.Upsample(size=(80, 112, 80), mode="trilinear", align_corners=True)
        self.upsample_64 = nn.Upsample(size=(40, 56, 40), mode="trilinear", align_corners=True)
        self.upsample_48 = nn.Upsample(size=(20, 28, 20), mode="trilinear", align_corners=True)
        self.upsample_32 = nn.Upsample(size=(20, 28, 20), mode="trilinear", align_corners=True)
        self.upsample_24 = nn.Upsample(size=(10, 14, 10), mode="trilinear", align_corners=True)

        # Output
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
        p1 = self.ads_96(e1)
        e2 = self.encoder2(p1)
        p2 = self.ads_64(e2)
        e3 = self.encoder3(p2)
        p3 = self.ads_48(e3)
        e4 = self.encoder4(p3)
        p4 = self.ads_32(e4)
        e5 = self.encoder5(p4)
        p5 = self.ads_24(e5)
        e6 = self.encoder6(p5)

        # Bottleneck
        bottleneck = self.bottleneck(e6)
        bottleneck2 = self.bottleneck2(torch.cat([e6, bottleneck], dim=1))

        # Decoding phase + skip connections
        up5 = self.upsample_32(bottleneck2)
        d5 = self.decoder5(torch.cat([e5, up5], dim=1))
        up4 = self.upsample_48(d5)
        d4 = self.decoder4(torch.cat([e4, up4], dim=1))
        up3 = self.upsample_64(d4)
        d3 = self.decoder3(torch.cat([e3, up3], dim=1))
        up2 = self.upsample_96(d3)
        d2 = self.decoder2(torch.cat([e2, up2], dim=1))
        up1 = self.upsample_128(d2)
        d1 = self.decoder1(torch.cat([e1, up1], dim=1))

        # Output
        output = self.output(d1)

        return output

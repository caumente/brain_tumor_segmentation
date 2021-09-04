import torch
from torch import nn
from layers import conv1x1
from layers import LevelBlock
from layers import ConvInNormLeReLU


class DeepUNet(nn.Module):
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

        self.outconv = conv1x1(widths[0] // 2, regions)

    def forward(self, x):
        # Encoding phase
        e1 = self.encoder1(x)
        d1 = self.ads_96(e1)
        e2 = self.encoder2(d1)
        d2 = self.ads_64(e2)
        e3 = self.encoder3(d2)
        d3 = self.ads_48(e3)
        e4 = self.encoder4(d3)
        d4 = self.ads_32(e4)
        e5 = self.encoder5(d4)
        d5 = self.ads_24(e5)
        e6 = self.encoder6(d5)

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
        output = self.outconv(d1)

        return output

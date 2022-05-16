import torch
from torch import nn
from torch.nn.functional import interpolate
from src.models.layers import conv1x1
from src.models.layers import LevelBlock
from src.models.layers import ConvInNormLeReLU
# from monai.transforms import Resize
# from torchvision.transforms import Resize
# from torchio.transforms import Resize
# from torchio.transforms import Resample


class ShallowUNetTestingDeepSupervision(nn.Module):
    """
    This class implements a variation of 3D Unet network. Main modifications are:
        - Replacement of ReLU activation layer by LeakyReLU
        - Use of instance normalization to ensure a normalization by each sequence

    """

    name = "Shallow U-Net"

    def __init__(self, sequences, regions, width, deep_supervision):
        super(ShallowUNetTestingDeepSupervision, self).__init__()

        self.deep_supervision = deep_supervision
        widths = [width * 2 ** i for i in range(4)]

        # Encoders
        self.encoder1 = LevelBlock(sequences, widths[0] // 2, widths[0])
        self.encoder2 = LevelBlock(widths[0] + 4, widths[1] // 2, widths[1])
        self.encoder3 = LevelBlock(widths[1] + 4, widths[2] // 2, widths[2])
        self.encoder4 = LevelBlock(widths[2] + 4, widths[3] // 2, widths[3])

        # Bottleneck
        self.bottleneck = LevelBlock(widths[3] + 4, widths[3], widths[3])
        self.bottleneck2 = ConvInNormLeReLU(widths[3] * 2 + 4, widths[2])

        # Decoders
        self.decoder3 = LevelBlock(widths[2] * 2 + 4, widths[2], widths[1])
        self.decoder2 = LevelBlock(widths[1] * 2 + 4, widths[1], widths[0])
        self.decoder1 = LevelBlock(widths[0] * 2 + 4, widths[0], widths[0] // 2)

        # Upsample, downsample and output steps
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.downsample = nn.MaxPool3d(2, 2)

        # # Resizer
        # self.resizer1 = Resample(target=(2, 2, 2), image_interpolation="linear")
        # self.resizer2 = Resample(target=(4, 4, 4), image_interpolation="linear")
        # self.resizer3 = Resample(target=(8, 8, 8), image_interpolation="linear")

        # Output
        if self.deep_supervision:
            self.output9 = nn.Sequential(
                conv1x1(widths[0], regions)
            )
            self.output8 = nn.Sequential(
                nn.ConvTranspose3d(widths[1], widths[0], kernel_size=2, stride=2),
                conv1x1(widths[0], regions)
            )
            self.output7 = nn.Sequential(
                nn.ConvTranspose3d(widths[2], widths[0], kernel_size=4, stride=4),
                conv1x1(widths[0], regions)
            )
            self.output6 = nn.Sequential(
                nn.Upsample(scale_factor=8, mode="trilinear"),
                ConvInNormLeReLU(widths[3], widths[0]),
                conv1x1(widths[0], regions)
            )
            self.output5 = nn.Sequential(
                nn.Upsample(scale_factor=8, mode="trilinear"),
                ConvInNormLeReLU(widths[3], widths[0]),
                conv1x1(widths[0], regions)
            )
            self.output4 = nn.Sequential(
                nn.Upsample(scale_factor=8, mode="trilinear"),
                ConvInNormLeReLU(widths[2], widths[0]),
                conv1x1(widths[0], regions)
            )
            self.output3 = nn.Sequential(
                nn.ConvTranspose3d(widths[1], widths[0], kernel_size=4, stride=4),
                conv1x1(widths[0], regions)
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

        # Sequences resized
        r1 = interpolate(input=x, scale_factor=(.5, .5, .5), mode="trilinear")
        r2 = interpolate(input=x, scale_factor=(.25, .25, .25), mode="trilinear")
        r3 = interpolate(input=x, scale_factor=(.125, .125, .125), mode="trilinear")

        # Encoding phase
        e1 = self.encoder1(x)
        p1 = self.downsample(e1)
        e2 = self.encoder2(torch.cat([p1, r1], dim=1))
        p2 = self.downsample(e2)
        e3 = self.encoder3(torch.cat([p2, r2], dim=1))
        p3 = self.downsample(e3)
        e4 = self.encoder4(torch.cat([p3, r3], dim=1))

        # Bottleneck phase
        bottleneck = self.bottleneck(torch.cat([e4, r3], dim=1))
        bottleneck2 = self.bottleneck2(torch.cat([e4, bottleneck, r3], dim=1))

        # Decoding phase + skip connections
        up3 = self.upsample(bottleneck2)
        d3 = self.decoder3(torch.cat([e3, up3, r2], dim=1))
        up2 = self.upsample(d3)
        d2 = self.decoder2(torch.cat([e2, up2, r1], dim=1))
        up1 = self.upsample(d2)
        d1 = self.decoder1(torch.cat([e1, up1, x], dim=1))

        # Output
        if self.deep_supervision:
            output9 = self.output9(e1)
            output8 = self.output8(e2)
            output7 = self.output7(e3)
            output6 = self.output6(e4)
            output5 = self.output5(bottleneck)
            output4 = self.output4(bottleneck2)
            output3 = self.output3(d3)
            output2 = self.output2(d2)
            output1 = self.output1(d1)

            return [output9, output8, output7, output6, output5, output4, output3, output2, output1]
        else:
            output1 = self.output1(d1)

            return output1


def test():
    seq_input = torch.rand(1, 4, 160, 224, 160)
    seq_ouput = torch.rand(1, 3, 160, 224, 160)

    model = ShallowUNetTestingDeepSupervision(sequences=4, regions=3, width=6, deep_supervision=True)
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

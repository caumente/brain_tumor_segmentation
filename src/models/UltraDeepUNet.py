import torch
from torch import nn
from src.models.layers import conv1x1
from src.models.layers import LevelBlock
#from src.utils.models import count_parameters


class UltraDeepUNet(nn.Module):
    """
    This class implements a variation of 3D Unet network. Main modifications are:
        - Replacement of ReLU activation layer by LeakyReLU
        - Use of instance normalization to ensure a normalization by each sequence

    """

    name = "Ultra Deep U-Net"

    def __init__(self, sequences, regions, width, deep_supervision=False):
        super(UltraDeepUNet, self).__init__()

        self.deep_supervision = deep_supervision
        widths = [width + (12*i) for i in range(10)]

        # Encoders
        self.encoder1 = LevelBlock(sequences, widths[0], widths[0])
        self.encoder2 = LevelBlock(widths[0], widths[1], widths[1])
        self.encoder3 = LevelBlock(widths[1], widths[2], widths[2])
        self.encoder4 = LevelBlock(widths[2], widths[3], widths[3])
        self.encoder5 = LevelBlock(widths[3], widths[4], widths[4])
        self.encoder6 = LevelBlock(widths[4], widths[5], widths[5])
        self.encoder7 = LevelBlock(widths[5], widths[6], widths[6])
        self.encoder8 = LevelBlock(widths[6], widths[7], widths[7])
        self.encoder9 = LevelBlock(widths[7], widths[8], widths[8])
        self.encoder10 = LevelBlock(widths[8], widths[9], widths[9])

        # Downsamplings
        self.ads1 = nn.AdaptiveMaxPool3d(output_size=(140, 196, 140))
        self.ads2 = nn.AdaptiveMaxPool3d(output_size=(120, 168, 120))
        self.ads3 = nn.AdaptiveMaxPool3d(output_size=(100, 140, 100))
        self.ads4 = nn.AdaptiveMaxPool3d(output_size=(80, 112, 80))
        self.ads5 = nn.AdaptiveMaxPool3d(output_size=(60, 84, 60))
        self.ads6 = nn.AdaptiveMaxPool3d(output_size=(40, 56, 40))
        self.ads7 = nn.AdaptiveMaxPool3d(output_size=(20, 28, 20))
        self.ads8 = nn.AdaptiveMaxPool3d(output_size=(10, 14, 10))
        self.ads9 = nn.AdaptiveMaxPool3d(output_size=(5, 7, 5))

        # Upsamplings
        self.upsample1 = nn.Upsample(size=(160, 224, 160), mode="trilinear", align_corners=True)
        self.upsample2 = nn.Upsample(size=(140, 196, 140), mode="trilinear", align_corners=True)
        self.upsample3 = nn.Upsample(size=(120, 168, 120), mode="trilinear", align_corners=True)
        self.upsample4 = nn.Upsample(size=(100, 140, 100), mode="trilinear", align_corners=True)
        self.upsample5 = nn.Upsample(size=(80, 112, 80), mode="trilinear", align_corners=True)
        self.upsample6 = nn.Upsample(size=(60, 84, 60), mode="trilinear", align_corners=True)
        self.upsample7 = nn.Upsample(size=(40, 56, 40), mode="trilinear", align_corners=True)
        self.upsample8 = nn.Upsample(size=(20, 28, 20), mode="trilinear", align_corners=True)
        self.upsample9 = nn.Upsample(size=(10, 14, 10), mode="trilinear", align_corners=True)


        # bottleneck
        self.bottleneck = LevelBlock(widths[9], widths[9], widths[9])

        # Decoders
        self.decoder10 = LevelBlock(2 * widths[9], widths[9], widths[8])
        self.decoder9 = LevelBlock(2*widths[8], widths[8], widths[7])
        self.decoder8 = LevelBlock(2*widths[7], widths[7], widths[6])
        self.decoder7 = LevelBlock(2*widths[6], widths[6], widths[5])
        self.decoder6 = LevelBlock(2*widths[5], widths[5], widths[4])
        self.decoder5 = LevelBlock(2*widths[4], widths[4], widths[3])
        self.decoder4 = LevelBlock(2*widths[3], widths[3], widths[2])
        self.decoder3 = LevelBlock(2*widths[2], widths[2], widths[1])
        self.decoder2 = LevelBlock(2*widths[1], widths[1], widths[0])
        self.decoder1 = LevelBlock(2*widths[0], widths[0], widths[0] // 2)

        # Output
        self.output4 = nn.Sequential(
            nn.Upsample(size=(160, 224, 160), mode="trilinear", align_corners=True),
            nn.Conv3d(widths[2], widths[2], kernel_size=(3, 3, 3), stride=1, padding=1),
            conv1x1(widths[2], regions)
        )
        self.output3 = nn.Sequential(
            nn.Upsample(size=(160, 224, 160), mode="trilinear", align_corners=True),
            nn.Conv3d(widths[1], widths[1], kernel_size=(3, 3, 3), stride=1, padding=1),
            conv1x1(widths[1], regions)
        )
        self.output2 = nn.Sequential(
            nn.Upsample(size=(160, 224, 160), mode="trilinear", align_corners=True),
            nn.Conv3d(widths[0], widths[0], kernel_size=(3, 3, 3), stride=1, padding=1),
            conv1x1(widths[0], regions)
        )
        self.output1 = conv1x1(widths[0] // 2, regions)


    def forward(self, x):
        # Encoding phase
        e1 = self.encoder1(x)
        p1 = self.ads1(e1)
        e2 = self.encoder2(p1)
        p2 = self.ads2(e2)
        e3 = self.encoder3(p2)
        p3 = self.ads3(e3)
        e4 = self.encoder4(p3)
        p4 = self.ads4(e4)
        e5 = self.encoder5(p4)
        p5 = self.ads5(e5)
        e6 = self.encoder6(p5)
        p6 = self.ads6(e6)
        e7 = self.encoder7(p6)
        p7 = self.ads7(e7)
        e8 = self.encoder8(p7)
        p8 = self.ads8(e8)
        e9 = self.encoder9(p8)
        p9 = self.ads9(e9)
        e10 = self.encoder10(p9)


        bottleneck = self.bottleneck(e10)
        bottleneck = self.decoder10(torch.cat([e10, bottleneck], dim=1))

        up9 = self.upsample9(bottleneck)
        d9 = self.decoder9(torch.cat([e9, up9], dim=1))
        up8 = self.upsample8(d9)
        d8 = self.decoder8(torch.cat([e8, up8], dim=1))
        up7 = self.upsample7(d8)
        d7 = self.decoder7(torch.cat([e7, up7], dim=1))
        up6 = self.upsample6(d7)
        d6 = self.decoder6(torch.cat([e6, up6], dim=1))
        up5 = self.upsample5(d6)
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
        if self.deep_supervision:
            output4 = self.output4(d4)
            output3 = self.output3(d3)
            output2 = self.output2(d2)
            output1 = self.output1(d1)

            return [output4, output3, output2, output1]
        else:
            output1 = self.output1(d1)

            return output1



def test():
    seq_input = torch.rand(1, 4, 160, 224, 160)
    seq_ouput = torch.rand(1, 3, 160, 224, 160)

    model = UltraDeepUNet(sequences=4, regions=3, width=6, deep_supervision=True)
    preds = model(seq_input)
    #print(count_parameters(model))

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

import torch
from torch import nn
from src.models.layers import conv1x1
from src.models.layers import LevelBlock
from src.models.layers import ConvInNormLeReLU


class MISU(nn.Module):
    """
    This class implements a variation of 3D Unet network. Main modifications are:
        - Replacement of ReLU activation layer by LeakyReLU
        - Use of instance normalization to ensure a normalization by each sequence

    """

    name = "Multi-Input Shallow U-Net"

    def __init__(self, sequences, regions, width, deep_supervision):
        super(MISU, self).__init__()

        self.deep_supervision = deep_supervision
        widths = [width * 2 ** i for i in range(4)]

        # Encoders 1
        self.encoder11_simple = LevelBlock(sequences//2, widths[0] // 2, widths[0])
        self.encoder12_simple = LevelBlock(sequences//2, widths[0] // 2, widths[0])
        self.encoder13_multiple = LevelBlock(sequences, widths[0] // 2, widths[0])
        # Encoders 2
        self.encoder21 = LevelBlock(widths[0], widths[1] // 2, widths[1])
        self.encoder22 = LevelBlock(widths[0], widths[1] // 2, widths[1])
        self.encoder23 = LevelBlock(widths[0], widths[1] // 2, widths[1])
        # Encoders 3
        self.encoder31 = LevelBlock(widths[1], widths[2] // 2, widths[2])
        self.encoder32 = LevelBlock(widths[1], widths[2] // 2, widths[2])
        self.encoder33 = LevelBlock(widths[1], widths[2] // 2, widths[2])
        # Encoders 3
        self.encoder41 = LevelBlock(widths[2], widths[3] // 2, widths[3])
        self.encoder42 = LevelBlock(widths[2], widths[3] // 2, widths[3])
        self.encoder43 = LevelBlock(widths[2], widths[3] // 2, widths[3])

        # Bottleneck
        self.bottleneck11 = LevelBlock(widths[3], widths[3], widths[3])
        self.bottleneck12 = LevelBlock(widths[3], widths[3], widths[3])
        self.bottleneck13 = LevelBlock(widths[3], widths[3], widths[3])
        self.bottleneck21 = ConvInNormLeReLU(widths[3] * 4, widths[2])
        self.bottleneck22 = ConvInNormLeReLU(widths[3] * 4, widths[2])
        self.bottleneck23 = ConvInNormLeReLU(widths[3] * 4, widths[2])

        # Decoders 3
        self.decoder31 = LevelBlock(widths[2] * 4, widths[2], widths[1])
        self.decoder32 = LevelBlock(widths[2] * 4, widths[2], widths[1])
        self.decoder33 = LevelBlock(widths[2] * 4, widths[2], widths[1])
        # Decoders 2
        self.decoder21 = LevelBlock(widths[1] * 4, widths[1], widths[0])
        self.decoder22 = LevelBlock(widths[1] * 4, widths[1], widths[0])
        self.decoder23 = LevelBlock(widths[1] * 4, widths[1], widths[0])
        # Decoders 1
        self.decoder11 = LevelBlock(widths[0] * 4, widths[0], widths[0] // 2)
        self.decoder12 = LevelBlock(widths[0] * 4, widths[0], widths[0] // 2)
        self.decoder13 = LevelBlock(widths[0] * 4, widths[0], widths[0] // 2)

        # Upsample, downsample and output steps
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.downsample = nn.MaxPool3d(2, 2)

        # Output
        self.output31 = nn.Sequential(
            nn.ConvTranspose3d(widths[1], widths[1], kernel_size=4, stride=4),
            LevelBlock(widths[1], regions, regions)
        )
        self.output21 = nn.Sequential(
            nn.ConvTranspose3d(widths[0], widths[0], kernel_size=2, stride=2),
            LevelBlock(widths[0], regions, regions)
        )
        self.output32 = nn.Sequential(
            nn.ConvTranspose3d(widths[1], widths[1], kernel_size=4, stride=4),
            LevelBlock(widths[1], regions, regions)
        )
        self.output22 = nn.Sequential(
            nn.ConvTranspose3d(widths[0], widths[0], kernel_size=2, stride=2),
            LevelBlock(widths[0], regions, regions)
        )
        self.output33 = nn.Sequential(
            nn.ConvTranspose3d(widths[1], widths[1], kernel_size=4, stride=4),
            LevelBlock(widths[1], regions, regions)
        )
        self.output23 = nn.Sequential(
            nn.ConvTranspose3d(widths[0], widths[0], kernel_size=2, stride=2),
            LevelBlock(widths[0], regions, regions)
        )

        self.output11 = LevelBlock(widths[0] // 2, widths[0] // 2, 2)
        self.output12 = LevelBlock(widths[0] // 2, widths[0] // 2, 1)
        self.output13 = LevelBlock(widths[0] // 2, widths[0] // 2, 3)

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
        e11 = self.encoder11_simple(x[:, :2, :, :, :])
        p11 = self.downsample(e11)
        e21 = self.encoder21(p11)
        p21 = self.downsample(e21)
        e31 = self.encoder31(p21)
        p31 = self.downsample(e31)
        e41 = self.encoder41(p31)

        bottleneck11 = self.bottleneck11(e41)

        # Encoding phase U-Net 2
        e12 = self.encoder12_simple(x[:, 2:, :, :, :])
        p12 = self.downsample(e12)
        e22 = self.encoder22(p12)
        p22 = self.downsample(e22)
        e32 = self.encoder32(p22)
        p32 = self.downsample(e32)
        e42 = self.encoder42(p32)

        bottleneck12 = self.bottleneck12(e42)

        # Encoding phase U-Net 3
        e13 = self.encoder13_multiple(x)
        p13 = self.downsample(e13)
        e23 = self.encoder23(p13)
        p23 = self.downsample(e23)
        e33 = self.encoder33(p23)
        p33 = self.downsample(e33)
        e43 = self.encoder43(p33)

        bottleneck13 = self.bottleneck13(e43)

        """
        Decodings
        """

        # Decoding phase + skip connections
        bottleneck21 = self.bottleneck21(torch.cat([e41, e42, e43, bottleneck11], dim=1))
        up31 = self.upsample(bottleneck21)
        d31 = self.decoder31(torch.cat([e31, e32, e33, up31], dim=1))
        up21 = self.upsample(d31)
        d21 = self.decoder21(torch.cat([e21, e22, e23, up21], dim=1))
        up11 = self.upsample(d21)
        d11 = self.decoder11(torch.cat([e11, e12, e13, up11], dim=1))

        if self.deep_supervision:
            output31 = self.output31(d31)
            output21 = self.output21(d21)
            output11 = self.output11(d11)

            output1 = [output31, output21, output11]
        else:
            output1 = self.output11(d11)

        # Decoding phase + skip connections
        bottleneck22 = self.bottleneck22(torch.cat([e41, e42, e43, bottleneck12], dim=1))
        up32 = self.upsample(bottleneck22)
        d32 = self.decoder32(torch.cat([e31, e32, e33, up32], dim=1))
        up22 = self.upsample(d32)
        d22 = self.decoder22(torch.cat([e21, e22, e23, up22], dim=1))
        up12 = self.upsample(d22)
        d12 = self.decoder12(torch.cat([e11, e12, e13, up12], dim=1))

        if self.deep_supervision:
            output32 = self.output32(d32)
            output22 = self.output22(d22)
            output12 = self.output12(d12)

            output2 = [output32, output22, output12]
        else:
            output2 = self.output12(d12)

        # Decoding phase + skip connections
        bottleneck23 = self.bottleneck23(torch.cat([e41, e42, e43, bottleneck13], dim=1))
        up33 = self.upsample(bottleneck23)
        d33 = self.decoder33(torch.cat([e31, e32, e33, up33], dim=1))
        up23 = self.upsample(d33)
        d23 = self.decoder23(torch.cat([e21, e22, e23, up23], dim=1))
        up13 = self.upsample(d23)
        d13 = self.decoder13(torch.cat([e11, e12, e13, up13], dim=1))

        if self.deep_supervision:
            output33 = self.output33(d33)
            output23 = self.output23(d23)
            output13 = self.output13(d13)

            output3 = [output33, output23, output13]
        else:
            output3 = self.output13(d13)

        """
        Results concatenation
        """

        if not self.deep_supervision:
            et = torch.unsqueeze(torch.add(output1[:, 0, :, :, :], output3[:, 0, :, :, :]), dim=1)
            tc = torch.unsqueeze(torch.add(output1[:, 1, :, :, :], output3[:, 1, :, :, :]), dim=1)
            wt = torch.unsqueeze(torch.add(output2[:, 0, :, :, :], output3[:, 2, :, :, :]), dim=1)

            concatenated_output = torch.concat([et, tc, wt], dim=1)

        else:
            concatenated_output = []
            for o1, o2, o3 in zip(output1, output2, output3):
                et = torch.unsqueeze(torch.add(o1[:, 0, :, :, :], o3[:, 0, :, :, :]), dim=1)
                tc = torch.unsqueeze(torch.add(o1[:, 1, :, :, :], o3[:, 1, :, :, :]), dim=1)
                wt = torch.unsqueeze(torch.add(o2[:, 0, :, :, :], o3[:, 2, :, :, :]), dim=1)

                concatenated_output.append(torch.concat([et, tc, wt], dim=1))

        return concatenated_output


def test():
    seq_input = torch.rand(1, 4, 160, 224, 160)
    seq_ouput = torch.rand(1, 3, 160, 224, 160)

    model = MISU(sequences=4, regions=3, width=6, deep_supervision=True)
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

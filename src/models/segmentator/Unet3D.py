import torch
from torch import nn
from src.models.layers import conv1x1
from src.models.layers import UBlock


class UNet3D(nn.Module):

    name = "3D U-Net"

    def __init__(self, sequences, regions):
        super(UNet3D, self).__init__()

        features = [64, 128, 256, 512]

        # Encoders
        self.encoder1 = UBlock(sequences, features[0] // 2, features[0])
        self.encoder2 = UBlock(features[0], features[1] // 2, features[1])
        self.encoder3 = UBlock(features[1], features[2] // 2, features[2])

        # Bottleneck
        self.bottleneck = UBlock(features[2], features[2], features[3])

        # Decoders
        self.decoder3 = UBlock(features[3] + features[2], features[2], features[2])
        self.decoder2 = UBlock(features[2] + features[1], features[1], features[1])
        self.decoder1 = UBlock(features[1] + features[0], features[0], features[0])

        # Upsample, downsample and output steps
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.downsample = nn.MaxPool3d(2, 2)
        self.output = conv1x1(features[0], regions)

    def forward(self, x):

        # Encoding phase
        e1 = self.encoder1(x)
        p1 = self.downsample(e1)
        e2 = self.encoder2(p1)
        p2 = self.downsample(e2)
        e3 = self.encoder3(p2)
        p3 = self.downsample(e3)

        # Bottleneck
        bottleneck = self.bottleneck(p3)

        # Decoder
        up3 = self.upsample(bottleneck)
        d3 = self.decoder3(torch.cat([e3, up3], dim=1))
        up2 = self.upsample(d3)
        d2 = self.decoder2(torch.cat([e2, up2], dim=1))
        up1 = self.upsample(d2)
        d1 = self.decoder1(torch.cat([e1, up1], dim=1))

        # Output
        out = self.output(d1)

        return out



def test():
    seq_input = torch.rand(1, 4, 160, 224, 160)
    seq_ouput = torch.rand(1, 3, 160, 224, 160)

    model = UNet3D(sequences=4, regions=3)
    preds = model(seq_input)

    print(seq_input.shape)
    print(preds.shape)

    assert seq_ouput.shape == preds.shape


if __name__ == "__main__":
    test()
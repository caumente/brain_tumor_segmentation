import torch
from torch import nn
from src.models.layers import conv1x1
from src.models.layers import UBlock


class UNet3D(nn.Module):

    name = "3D U-Net"

    def __init__(self, sequences, regions):
        super(UNet3D, self).__init__()

        features = [64, 128, 256, 256]

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
        # self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.upsample3 = nn.ConvTranspose3d(in_channels=features[3], out_channels=features[3], kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose3d(in_channels=features[2], out_channels=features[2], kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose3d(in_channels=features[1], out_channels=features[1], kernel_size=2, stride=2)
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
        up3 = self.upsample3(bottleneck)
        d3 = self.decoder3(torch.cat([e3, up3], dim=1))
        up2 = self.upsample2(d3)
        d2 = self.decoder2(torch.cat([e2, up2], dim=1))
        up1 = self.upsample1(d2)
        d1 = self.decoder1(torch.cat([e1, up1], dim=1))

        # Output
        out = self.output(d1)

        return out


if __name__ == "__main__":
    seq_input = torch.rand(1, 4, 160, 224, 160)
    # seq_ouput = torch.rand(1, 3, 160, 224, 160)

    model = UNet3D(sequences=4, regions=3)
    preds = model(seq_input)
    from flopth import flopth
    flops, params = flopth(model, in_size=((4, 160, 224, 160),), show_detail=False, bare_number=True)
    print(flops/1000000000000)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))
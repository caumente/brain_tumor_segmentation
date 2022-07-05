import torch
from torch import nn
from src.models.layers import conv1x1
from src.models.layers import LevelBlock
from src.models.layers import ConvInNormLeReLU


class ShallowUNet_v3(nn.Module):
    """
    This class implements a variation of 3D Unet network. Main modifications are:
        - Replacement of ReLU activation layer by LeakyReLU
        - Use of instance normalization to ensure a normalization by each sequence

    """

    name = "Shallow U-Net V2"

    def __init__(self, sequences, regions, width, deep_supervision):
        super(ShallowUNet_v3, self).__init__()

        self.deep_supervision = deep_supervision
        widths = [width * 2 ** i for i in range(4)]

        # Encoders
        self.encoder1 = LevelBlock(sequences, widths[0] // 2, widths[1])
        self.encoder2 = LevelBlock(widths[1], widths[2], widths[3])
        self.encoder3 = LevelBlock(widths[3], widths[3], widths[3])


        # Decoders
        self.decoder3 = LevelBlock(widths[3] * 2, widths[3], widths[3])
        self.decoder2 = LevelBlock(widths[3] * 2, widths[2], widths[1])
        self.decoder1 = LevelBlock(widths[1] * 2, widths[0], widths[0] // 2)

        # Upsample, downsample and output steps
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.upsample_ = nn.ConvTranspose3d(widths[3], widths[3], kernel_size=4, stride=1)
        self.downsample = nn.MaxPool3d(2, 2)
        self.downsample_ = nn.MaxPool3d(5, 1)

        self.dense1 = nn.Linear(320 * width, 16 * width)
        self.dropout = nn.Dropout(.3)
        self.dense2 = nn.Linear(16 * width, 320 * width)
        # self.upsample3 = nn.ConvTranspose3d(widths[2], widths[2], kernel_size=2, stride=2)
        # self.upsample2 = nn.ConvTranspose3d(widths[1], widths[1], kernel_size=2, stride=2)
        # self.upsample1 = nn.ConvTranspose3d(widths[0], widths[0], kernel_size=2, stride=2)
        # self.downsample1 = nn.Conv3d(widths[1], widths[0], kernel_size=(2, 2, 2), stride=2)
        # self.downsample2 = nn.Conv3d(widths[2], widths[1], kernel_size=(2, 2, 2), stride=2)
        # self.downsample3 = nn.Conv3d(widths[3], widths[2], kernel_size=(2, 2, 2), stride=2)

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(320 * width, 16 * width),
            nn.Dropout(.3),
            nn.Linear(16 * width, 320 * width),
            nn.Unflatten(1, torch.Size([widths[3], 2, 10, 2]))
        )

        # Output
        if self.deep_supervision:
            self.output2 = nn.Sequential(
                nn.ConvTranspose3d(widths[1], widths[0], kernel_size=2, stride=2),
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
        # print(e2.shape)
        p2 = self.downsample(e2)

        e3_ = p2
        # print(p2.shape)
        for i in range(9):
            p3 = self.downsample_(e3_)
            e3_ = self.encoder3(p3)

            # print(e3_.shape)

        p4 = self.downsample(e3_)
        # print(p4.shape)
        # print("bottleneck")
        bottleneck = self.bottleneck(p4)
        # print(bottleneck.shape)

        up3 = self.upsample(bottleneck)
        # print(up3.shape)

        up2 = up3
        for i in range(12):
            up2 = self.upsample_(up2)
            up2 = self.encoder3(up2)
            # print(up2.shape)

        up2 = self.upsample(up2)
        # print(up2.shape)
        d2 = self.decoder2(torch.cat([e2, up2], dim=1))
        #print(d2.shape)
        up1 = self.upsample(d2)
        #print(up1.shape)
        d1 = self.decoder1(torch.cat([e1, up1], dim=1))
        #print(d1.shape)


        # Output
        if self.deep_supervision:
            output2 = self.output2(d2)
            output1 = self.output1(d1)

            return [output2, output1]
        else:
            output1 = self.output1(d1)

            return output1



def __test():
    seq_input = torch.rand(1, 4, 160, 224, 160)
    seq_ouput = torch.rand(1, 3, 160, 224, 160)

    model = ShallowUNet_v3(sequences=4, regions=3, width=12, deep_supervision=True)
    #print(count_parameters(model))
    preds = model(seq_input)

    # print(seq_input.shape)
    # if model.deep_supervision:
    #     for p in preds:
    #         print(p.shape)
    #         assert seq_ouput.shape == p.shape
    # else:
    #     print(preds.shape)
    #     assert seq_ouput.shape == preds.shape


if __name__ == "__main__":
    __test()

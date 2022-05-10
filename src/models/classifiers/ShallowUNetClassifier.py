import torch
from torch import nn
from src.models.layers import LevelBlock
from src.models.layers import ConvInNormLeReLU
from src.models.layers import FullyConnectedClassifier


class ShallowUNetClassifier(nn.Module):
    """
    This class implements the encoder from the Shallow UNet network to create a classifier.
    """

    name = "Shallow U-Net ShallowUNetClassifier"

    def __init__(self, sequences, classes, width, dense_neurons):
        super(ShallowUNetClassifier, self).__init__()

        widths = [width * 2 ** i for i in range(5)]

        # Encoders
        self.encoder1 = LevelBlock(sequences, widths[0] // 2, widths[0])
        self.encoder2 = LevelBlock(widths[0], widths[1] // 2, widths[1])
        self.encoder3 = LevelBlock(widths[1], widths[2] // 2, widths[2])
        self.encoder4 = LevelBlock(widths[2], widths[3] // 2, widths[3])
        self.encoder5 = LevelBlock(widths[3], widths[4] // 2, widths[4])

        # Bottleneck
        # self.bottleneck = LevelBlock(widths[4], widths[4], widths[4])
        # self.bottleneck2 = ConvInNormLeReLU(widths[3] * 2, widths[2])

        # Upsample, downsample and output steps
        self.downsample = nn.MaxPool3d(2, 2)

        # FCN
        self.classifier = FullyConnectedClassifier(width, dense_neurons, classes)

        self.weights_initialization()

    def weights_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoding phase
        x = self.encoder1(x)
        x = self.downsample(x)
        x = self.encoder2(x)
        x = self.downsample(x)
        x = self.encoder3(x)
        x = self.downsample(x)
        x = self.encoder4(x)
        x = self.downsample(x)
        x = self.encoder5(x)

        # Bottleneck phase
        # x = self.bottleneck(x_)
        # x = self.bottleneck2(torch.cat([x_, x], dim=1))
        x = torch.flatten(x, 1)

        # FCN
        x = self.classifier(x)
        # x = torch.sigmoid(x)

        return x


def test():
    seq_input = torch.rand(1, 4, 160, 224, 160)

    model = ShallowUNetClassifier(sequences=4, classes=2, width=6, dense_neurons=128)
    preds = model(seq_input)
    print(preds.shape)


if __name__ == "__main__":
    test()

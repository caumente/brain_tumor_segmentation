import torch
from torch import nn
from src.models.layers import ConvInNormLeReLU
from src.models.layers import FullyConnectedClassifier


class BTSUNetClassifier(nn.Module):
    """
    This class implements the encoder from the BTS UNet network to create a classifier.
    """

    name = "BTS U-Net Classifier"

    def __init__(self, sequences, classes, width, dense_neurons):
        super(BTSUNetClassifier, self).__init__()

        widths = [width * 2 ** i for i in range(5)]
        # Encoders
        self.encoder1 = ConvInNormLeReLU(sequences, widths[0])
        self.encoder2 = ConvInNormLeReLU(widths[0], widths[1])
        self.encoder3 = ConvInNormLeReLU(widths[1], widths[2])
        self.encoder4 = ConvInNormLeReLU(widths[2], widths[3])
        self.encoder5 = ConvInNormLeReLU(widths[3], widths[4])

        # Upsample, downsample and output steps
        self.downsample = nn.MaxPool3d(2, 2)

        # FCN
        self.classifier = FullyConnectedClassifier(widths[-1], dense_neurons, classes)

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
        x = self.downsample(x)

        x = torch.flatten(x, 1)

        # FCN
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    seq_input = torch.rand(1, 4, 160, 224, 160)

    model = BTSUNetClassifier(sequences=4, classes=2, width=6, dense_neurons=128)
    preds = model(seq_input)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(preds.shape)

import torch
import torch.nn as nn

from resnet import ResNetEncoder, ResnetDecoder, ResNetBasicBlock, ResNetBottleNeckBlock


class SiameseResNet(nn.Module):

    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels * 2, n_classes)

    def forward(self, input1, input2):
        resnet_out1 = self.encoder(input1)
        resnet_out2 = self.encoder(input2)

        concat = torch.cat((resnet_out1, resnet_out2), 1)

        out = self.decoder(concat)

        return out


blocks_sizes = [32, 64, 128, 256]


def siamese_resnet18(in_channels, n_classes):
    return SiameseResNet(
        in_channels, n_classes, block=ResNetBasicBlock, deepths=[2, 2, 2, 2], blocks_sizes=blocks_sizes)


def siamese_resnet34(in_channels, n_classes):
    return SiameseResNet(
        in_channels, n_classes, block=ResNetBasicBlock, deepths=[3, 4, 6, 3], blocks_sizes=blocks_sizes)


def siamese_resnet50(in_channels, n_classes):
    return SiameseResNet(
        in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3], blocks_sizes=blocks_sizes
    )


def siamese_resnet101(in_channels, n_classes):
    return SiameseResNet(
        in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 23, 3], blocks_sizes=blocks_sizes
    )


def siamese_resnet152(in_channels, n_classes):
    return SiameseResNet(
        in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 8, 36, 3], blocks_sizes=blocks_sizes)


if __name__ == '__main__':
    from torchsummary import summary

    model = siamese_resnet50(1, 7)
    summary(model.cpu(), [(1, 500, 750), (1, 500, 750)])

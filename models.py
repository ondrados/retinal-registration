import torch
import torch.nn as nn

from resnet import ResNetEncoder, ResnetDecoder, ResNetBasicBlock, ResNetBottleNeckBlock


class CorrelationLayer(nn.Module):
    # https://github.com/limacv/CorrelationLayer/blob/master/correlation_torch.py
    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1):
        assert kernel_size == 1, "kernel_size other than 1 is not implemented"
        assert pad_size == max_displacement
        assert stride1 == stride2 == 1
        super().__init__()
        # self.pad_size = pad_size
        # self.kernel_size = kernel_size
        # self.stride1 = stride1
        # self.stride2 = stride2
        self.max_hdisp = max_displacement
        self.padlayer = nn.ConstantPad2d(pad_size, 0)
        self.out_size = (2 * self.max_hdisp + 1)**2

    def forward(self, in1, in2):
        in2_pad = self.padlayer(in2)
        offsety, offsetx = torch.meshgrid([torch.arange(0, 2 * self.max_hdisp + 1),
                                           torch.arange(0, 2 * self.max_hdisp + 1)])
        hei, wid = in1.shape[2], in1.shape[3]
        output = torch.cat([
            torch.mean(in1 * in2_pad[:, :, dy:dy+hei, dx:dx+wid], 1, keepdim=True)
            for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))
        ], 1)
        return output


class SiameseResNet(nn.Module):

    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        # self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels * 2, n_classes)
        self.correlation = CorrelationLayer()
        self.fc = ResnetDecoder(self.correlation.out_size, n_classes)

    def forward(self, input1, input2):
        resnet_out1 = self.encoder(input1)
        resnet_out2 = self.encoder(input2)
        # concat = torch.cat((resnet_out1, resnet_out2), 1)
        corr_out = self.correlation(resnet_out1, resnet_out2)

        out = self.fc(corr_out)

        return out


blocks_sizes = [64, 128, 256, 512]


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

    model = siamese_resnet18(1, 7)
    summary(model.cpu(), [(1, 500, 750), (1, 500, 750)])

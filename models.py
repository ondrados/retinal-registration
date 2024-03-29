import torch
import torch.nn as nn

from resnet import ResNetEncoder, ResnetDecoder, ResNetBasicBlock, ResNetBottleNeckBlock


def featureL2Norm(feature):
    epsilon = 1e-6
    #        print(feature.size())
    #        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)


class CorrelationLayer(nn.Module):
    # https://github.com/limacv/CorrelationLayer/blob/master/correlation_torch.py
    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1):
        super().__init__()
        self.max_hdisp = max_displacement
        self.padlayer = nn.ConstantPad2d(pad_size, 0)
        self.out_size = (2 * self.max_hdisp + 1) ** 2

    def forward(self, in1, in2):
        in2_pad = self.padlayer(in2)
        offsety, offsetx = torch.meshgrid([torch.arange(0, 2 * self.max_hdisp + 1),
                                           torch.arange(0, 2 * self.max_hdisp + 1)])
        hei, wid = in1.shape[2], in1.shape[3]
        output = torch.cat([
            torch.mean(in1 * in2_pad[:, :, dy:dy + hei, dx:dx + wid], 1, keepdim=True)
            for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))
        ], 1)
        return output


class FeatureCorrelation(torch.nn.Module):
    def __init__(self, normalization=True, matching_type='correlation'):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.matching_type = matching_type
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        if self.matching_type == 'correlation':
            # reshape features for matrix multiplication
            feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
            feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
            # perform matrix mult.
            feature_mul = torch.bmm(feature_B, feature_A)
            # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)

            if self.normalization:
                correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))

            return correlation_tensor

        if self.matching_type == 'subtraction':
            return feature_A.sub(feature_B)

        if self.matching_type == 'concatenation':
            return torch.cat((feature_A, feature_B), 1)


class FeatureRegression(nn.Module):
    def __init__(self, output_dim=7, batch_normalization=True, kernel_sizes=[7, 5, 5], channels=[384, 128, 64]):
        super().__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers - 1):  # last layer is linear
            k_size = kernel_sizes[i]
            ch_in = channels[i]
            ch_out = channels[i + 1]
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=k_size, padding=0))
            if batch_normalization:
                nn_modules.append(nn.BatchNorm2d(ch_out))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)
        self.linear = nn.Linear(5376, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear(x)
        return x


class InitialSiameseResNet(nn.Module):

    def __init__(self, in_channels, n_classes, matching_type='concatenation', *args, **kwargs):
        super().__init__()
        self.matching_type = matching_type
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)

        if self.matching_type == 'concatenation':
            self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels * 2, n_classes)
        elif self.matching_type == 'subtraction':
            self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        elif self.matching_type == 'correlation':
            self.correlation = CorrelationLayer()
            self.fc = ResnetDecoder(self.correlation.out_size, n_classes)

    def forward(self, input1, input2):
        resnet_out1 = self.encoder(input1)
        resnet_out2 = self.encoder(input2)

        if self.matching_type == 'concatenation':
            match = torch.cat((resnet_out1, resnet_out2), 1)
        elif self.matching_type == 'subtraction':
            match = torch.subtract(resnet_out1, resnet_out2)
        elif self.matching_type == 'correlation':
            match = self.correlation(resnet_out1, resnet_out2)
            return self.fc(match)

        out = self.decoder(match)

        return out


class SiameseResNet(nn.Module):

    def __init__(self, in_channels, n_classes, matching_type, channels, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.correlation = FeatureCorrelation(matching_type=matching_type)
        self.regression = FeatureRegression(output_dim=n_classes, channels=channels)

    def forward(self, input1, input2):
        resnet_out1 = self.encoder(input1)
        resnet_out2 = self.encoder(input2)

        corr_out = self.correlation(resnet_out1, resnet_out2)

        out = self.regression(corr_out)

        return out


blocks_sizes = [64, 128, 256, 512]


def siamese_resnet18(in_channels, n_classes, matching_type, channels):
    return SiameseResNet(
        in_channels, n_classes, matching_type, channels, block=ResNetBasicBlock, deepths=[2, 2, 2, 2],
        blocks_sizes=blocks_sizes)


def initial_siamese_resnet18(in_channels, n_classes, matching_type):
    return InitialSiameseResNet(
        in_channels, n_classes, matching_type, block=ResNetBasicBlock, deepths=[2, 2, 2, 2], blocks_sizes=blocks_sizes)


def siamese_resnet34(in_channels, n_classes, matching_type):
    return SiameseResNet(
        in_channels, n_classes, matching_type, block=ResNetBasicBlock, deepths=[3, 4, 6, 3], blocks_sizes=blocks_sizes)


def siamese_resnet50(in_channels, n_classes, matching_type):
    return SiameseResNet(
        in_channels, n_classes, matching_type, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3],
        blocks_sizes=blocks_sizes
    )


if __name__ == '__main__':
    from torchsummary import summary

    # model = initial_siamese_resnet18(1, 7, "concatenation")
    # model = initial_siamese_resnet18(1, 7, "subtraction")
    model = initial_siamese_resnet18(1, 7, "correlation")
    # model = siamese_resnet18(1, 7, "concatenation", channels=[1024, 128, 64])
    # model = siamese_resnet18(1, 7, "subtraction", channels=[512, 128, 64])
    # model = siamese_resnet18(1, 7, "correlation", channels=[384, 128, 64])
    summary(model.cpu(), [(1, 500, 750), (1, 500, 750)])

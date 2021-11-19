import torch
from torch import nn
import torch.nn.functional as F


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization, activation, kernel_size=4, stride=2, padding=1):
        super(Conv2DBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        if normalization == nn.utils.spectral_norm:
            self.norm = normalization(self.conv2d)
        else:
            self.norm = normalization
        self.activation = activation

    def forward(self, x):
        if self.norm == nn.utils.spectral_norm:
            return self.activation(self.norm(x))
        else:
            return self.activation(self.norm(self.conv2d(x)))


class Conv2DTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization, activation, kernel_size=4, stride=2, padding=1):
        super(Conv2DTransposeBlock, self).__init__()
        self.deconv2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = normalization
        self.activation = activation

    def forward(self, x):
        return self.activation(self.norm(self.deconv2d(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.res = nn.Sequential(
            Conv2DBlock(channels, channels, nn.InstanceNorm2d(num_features=channels),
                        nn.LeakyReLU(0.2, inplace=True), 3, 1, 1),
            Conv2DBlock(channels, channels, nn.InstanceNorm2d(num_features=channels),
                        nn.Identity(), 3, 1, 1)
        )

    def forward(self, x):
        return F.leaky_relu(x + self.res(x), 0.2, inplace=True)


class AgeEstimationNetwork(nn.Module):
    def __init__(self, age_group, in_channels=3, channels=[64, 128, 256, 512, 512, 512]):
        super(AgeEstimationNetwork, self).__init__()
        # assuming input shape 3 * 256 * 256
        layers = []
        for ch in channels:
            layers.append(Conv2DBlock(in_channels, ch, nn.BatchNorm2d(num_features=ch), nn.ReLU(True),
                                      4, 2, 1))
            in_channels = ch

        layers.append(nn.Flatten())
        layers.append(nn.Linear(channels[-1] * 4 * 4, 101))

        self.age_classifier = nn.Sequential(*layers)
        self.group_classifier = nn.Linear(101, age_group)

    def forward(self, x):
        age_logit = self.age_classifier(F.hardtanh(x))
        group_logit = self.group_classifier(age_logit)
        return age_logit, group_logit


class SubGenerator(nn.Module):
    def __init__(self, in_channels=3, channels=[32, 64, 128], n_residuals=4):
        super(SubGenerator, self).__init__()
        layers = []
        layers.append(Conv2DBlock(in_channels, channels[0], nn.InstanceNorm2d(num_features=channels[0]),
                                  nn.LeakyReLU(0.2, inplace=True), 9, 1, 4,))
        layers.append(Conv2DBlock(channels[0], channels[1], nn.InstanceNorm2d(num_features=channels[1]),
                                  nn.LeakyReLU(0.2, inplace=True), 4, 2, 1))
        layers.append(Conv2DBlock(channels[1], channels[2], nn.InstanceNorm2d(num_features=channels[2]),
                                  nn.LeakyReLU(0.2, inplace=True), 4, 2, 1,))
        # output shape will be 128 * 64 * 64

        for i in range(n_residuals):
            layers.append(ResidualBlock(channels[2]))

        layers.append(Conv2DTransposeBlock(channels[2], channels[1], nn.InstanceNorm2d(num_features=channels[1]),
                                           nn.LeakyReLU(0.2, inplace=True), 4, 2, 1))
        layers.append(Conv2DTransposeBlock(channels[1], channels[0], nn.InstanceNorm2d(num_features=channels[0]),
                                           nn.LeakyReLU(0.2, inplace=True), 4, 2, 1))
        layers.append(nn.Conv2d(channels[0], in_channels, 9, 1, 4))

        self.subgen = nn.Sequential(*layers)

    def forward(self, x):
        return self.subgen(x)


class Generator(nn.Module):
    def __init__(self, age_group):
        super(Generator, self).__init__()
        self.age_group = age_group
        self.gen = nn.ModuleList()
        for i in range(self.age_group - 1):
            self.gen.append(SubGenerator())

    def pfa_encoding(self, source, target):
        source, target = source.long(), target.long()
        code = torch.zeros((source.shape[0], self.age_group - 1, 1, 1, 1)).to(source)
        for i in range(source.shape[0]):
            code[i, source[i]: target[i], ...] = 1
        return code

    def forward(self, x, source, target):
        condition = self.pfa_encoding(source, target).to(x)
        for i in range(self.age_group - 1):
            x = x + self.gen[i](x) * condition[:, i]
        return x


class Discriminator(nn.Module):
    def __init__(self, age_group, channels=[64, 128, 256, 512, 512]):
        super(Discriminator, self).__init__()
        self.age_group = age_group
        self.conv = nn.Conv2d(3, channels[0], 4, 2, 1)

        layers = []
        in_channels = channels[0] + self.age_group
        for ch in channels[1: len(channels) - 1]:
            layers.append(Conv2DBlock(in_channels, ch, nn.BatchNorm2d(num_features=ch),
                                      nn.LeakyReLU(0.2, inplace=True), 4, 2, 1))
            in_channels = ch

        layers.append(Conv2DBlock(channels[-2], channels[-1], nn.BatchNorm2d(num_features=ch),
                                  nn.LeakyReLU(0.2, inplace=True), 4, 1, 1))
        layers.append(nn.Conv2d(channels[-1], 1, 4, 1, 1))

        self.disc = nn.Sequential(*layers)

    def group2onehot(self, group):
        code = torch.eye(self.age_group)[group.squeeze()]
        if len(code.size()) > 1:
            return code
        return code.unsqueeze(0)

    def group2feature(self, group, feature_size):
        onehot = self.group2onehot(group)
        return onehot.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, feature_size, feature_size)

    def forward(self, x, condition):
        x = F.leaky_relu(self.conv(x), 0.2, inplace=True)
        condition = self.group2feature(condition, x.shape[2]).to(x.device)
        return self.disc(torch.cat([x, condition], dim=1))

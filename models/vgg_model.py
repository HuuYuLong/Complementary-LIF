import torch
from spikingjelly.clock_driven import layer

__all__ = [
    'vggsnn', 'snn5', 'snn5_noAP'
]

from torch import nn


class SNN5(nn.Module):
    def __init__(self, neuron, num_classes=10, dropout=0.0, **kwargs):
        super(SNN5, self).__init__()
        pool = nn.Sequential(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            Layer(3, 16, 3, 1, 1, neuron, **kwargs),
            Layer(16, 64, 5, 1, 1, neuron, **kwargs),
            pool,
            Layer(64, 128, 5, 1, 1, neuron, **kwargs),
            pool,
            Layer(128, 256, 5, 1, 1, neuron, **kwargs),
            pool,
            Layer(256, 512, 3, 1, 1, neuron, **kwargs),
            pool,
        )
        W = int(32 / 2 / 2 / 2 / 2 / 2)

        self.classifier = nn.Linear(512 * W * W, num_classes)
        self.drop = layer.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        # print(x.shape)
        x = self.drop(torch.flatten(x, start_dim=-3, end_dim=-1))
        x = self.classifier(x)
        return x


# use for Figure.2
class SNN5_noAP(nn.Module):
    def __init__(self, neuron, num_classes=10, dropout=0.0, **kwargs):
        super(SNN5_noAP, self).__init__()
        pool = nn.Sequential(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(3, 16, 3, 1, 1, neuron, **kwargs),
            Layer(16, 64, 5, 2, 1, neuron, **kwargs),
            Layer(64, 128, 5, 2, 1, neuron, **kwargs),
            Layer(128, 256, 5, 4, 1, neuron, **kwargs),
            Layer(256, 256, 3, 2, 1, neuron, **kwargs),
        )
        # W = int(32 / 2 / 2 / 2 / 4 /  2)
        # if "fc_hw" in kwargs:
        #     W = int(kwargs["fc_hw"] / 2 / 2 / 2 / 2 / 2)

        self.classifier = nn.Linear(256, num_classes)
        self.drop = layer.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = self.drop(torch.flatten(x, start_dim=-3, end_dim=-1))
        x = self.classifier(x)
        return x


def snn5(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SNN5(neuron=neuron, num_classes=num_classes, dropout=neuron_dropout, **kwargs)


def snn5_noAP(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SNN5_noAP(neuron=neuron, num_classes=num_classes, dropout=neuron_dropout, **kwargs)


class Layer(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, neuron, **kwargs):
        super(Layer, self).__init__()
        self.fwd = nn.Sequential(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane)
        )
        self.act = neuron(**kwargs)

    def forward(self, x):
        x = self.fwd(x)
        x = self.act(x)
        # print(x.shape)
        return x


class VGGSNN(nn.Module):
    def __init__(self, neuron, num_classes=10, neuron_dropout=0.0, **kwargs):
        super(VGGSNN, self).__init__()
        pool = nn.Sequential(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1, neuron, **kwargs),
            Layer(64, 128, 3, 1, 1, neuron, **kwargs),
            pool,
            Layer(128, 256, 3, 1, 1, neuron, **kwargs),
            Layer(256, 256, 3, 1, 1, neuron, **kwargs),
            pool,
            Layer(256, 512, 3, 1, 1, neuron, **kwargs),
            Layer(512, 512, 3, 1, 1, neuron, **kwargs),
            pool,
            Layer(512, 512, 3, 1, 1, neuron, **kwargs),
            Layer(512, 512, 3, 1, 1, neuron, **kwargs),
            pool,
        )
        W = int(48 / 2 / 2 / 2 / 2)
        if "fc_hw" in kwargs:
            W = int(kwargs["fc_hw"] / 2 / 2 / 2 / 2)
        # self.T = 4
        # self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))
        self.classifier = nn.Linear(512 * W * W, num_classes)
        self.drop = layer.Dropout(neuron_dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        # x = torch.flatten(x, 2)
        x = self.drop(torch.flatten(x, start_dim=-3, end_dim=-1))
        x = self.classifier(x)
        return x


class VGGSNNwoAP(nn.Module):
    def __init__(self, neuron, num_classes=10, neuron_dropout=0.0, **kwargs):
        super(VGGSNNwoAP, self).__init__()
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1, neuron, **kwargs),
            Layer(64, 128, 3, 2, 1, neuron, **kwargs),
            Layer(128, 256, 3, 1, 1, neuron, **kwargs),
            Layer(256, 256, 3, 2, 1, neuron, **kwargs),
            Layer(256, 512, 3, 1, 1, neuron, **kwargs),
            Layer(512, 512, 3, 2, 1, neuron, **kwargs),
            Layer(512, 512, 3, 1, 1, neuron, **kwargs),
            Layer(512, 512, 3, 2, 1, neuron, **kwargs),
        )
        W = int(48 / 2 / 2 / 2 / 2)
        if "fc_hw" in kwargs:
            W = int(kwargs["fc_hw"] / 2 / 2 / 2 / 2)

        # self.T = 4
        # self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))
        self.classifier = nn.Linear(512 * W * W, num_classes)
        self.drop = layer.Dropout(neuron_dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # print(input.shape)
        x = self.features(input)
        # print(x.shape)
        x = self.drop(torch.flatten(x, start_dim=-3, end_dim=-1))

        x = self.classifier(x)
        return x


def vggsnn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return VGGSNN(neuron=neuron, num_classes=num_classes, dropout=neuron_dropout, **kwargs)


if __name__ == '__main__':
    # model = VGGSNNwoAP()
    from modules.neuron import ComplementaryLIFNeuron
    from thop import profile

    model = snn5_noAP(neuron=ComplementaryLIFNeuron)
    input = torch.randn(1, 3, 32, 32)
    flops, params = profile(model, inputs=(input,))
    print(model)

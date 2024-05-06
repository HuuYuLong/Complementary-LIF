import torch.nn as nn
from spikingjelly.clock_driven import layer

__all__ = [
    'PreActResNet', 'spiking_resnet18', 'spiking_resnet34', 'spiking_resnet50', 'spiking_resnet101', 'spiking_resnet152'
]


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, dropout, neuron: callable = None, **kwargs):
        super(PreActBlock, self).__init__()
        whether_bias = True
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=whether_bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = layer.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=3, stride=1, padding=1,
                               bias=whether_bias)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride,
                                      padding=0, bias=whether_bias)
        else:
            self.shortcut = nn.Sequential()

        self.relu1 = neuron(**kwargs)
        self.relu2 = neuron(**kwargs)

    def forward(self, x):
        x = self.relu1(self.bn1(x))
        out = self.conv1(x)
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        out = out + self.shortcut(x)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, dropout, neuron: callable = None, **kwargs):
        super(PreActBottleneck, self).__init__()
        whether_bias = True

        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=whether_bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=whether_bias)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = layer.Dropout(dropout)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, stride=1, padding=0,
                               bias=whether_bias)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride,
                                      padding=0, bias=whether_bias)
        else:
            self.shortcut = nn.Sequential()

        self.relu1 = neuron(**kwargs)
        self.relu2 = neuron(**kwargs)
        self.relu3 = neuron(**kwargs)

    def forward(self, x):
        x = self.relu1(self.bn1(x))

        out = self.conv1(x)
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.conv3(self.dropout(self.relu3(self.bn3(out))))

        out = out + self.shortcut(x)

        return out


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes, dropout, neuron: callable = None, **kwargs):
        super(PreActResNet, self).__init__()
        self.num_blocks = num_blocks

        self.data_channels = kwargs.get('c_in', 3)
        self.init_channels = 64
        self.conv1 = nn.Conv2d(self.data_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, dropout, neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, dropout, neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, dropout, neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, dropout, neuron, **kwargs)

        self.bn1 = nn.BatchNorm2d(512 * block.expansion)
        self.pool = nn.AvgPool2d(4)
        self.flat = nn.Flatten()
        self.drop = layer.Dropout(dropout)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.relu1 = neuron(**kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout, neuron, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.init_channels, out_channels, stride, dropout, neuron, **kwargs))
            self.init_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(self.relu1(self.bn1(out)))
        out = self.drop(self.flat(out))
        out = self.linear(out)
        return out


# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_planes, planes, stride=1, bn_type='', **kwargs_spikes):
#         super(Bottleneck, self).__init__()
#         self.kwargs_spikes = kwargs_spikes
#         self.nb_steps = kwargs_spikes['nb_steps']
#         self.conv1 = tdLayer(nn.Conv2d(in_planes, planes, kernel_size=1, bias=False), self.nb_steps)
#         self.bn1 = warpBN(planes, bn_type, self.nb_steps)
#         self.spike1 = LIFLayer(**kwargs_spikes)
#         self.conv2 = tdLayer(nn.Conv2d(planes, planes, kernel_size=3,
#                                        stride=stride, padding=1, bias=False), self.nb_steps)
#         self.bn2 = warpBN(planes, bn_type, self.nb_steps)
#         self.spike2 = LIFLayer(**kwargs_spikes)
#         self.conv3 = tdLayer(nn.Conv2d(planes, self.expansion *
#                                        planes, kernel_size=1, bias=False), self.nb_steps)
#         self.bn3 = warpBN(self.expansion *
#                           planes, bn_type, self.nb_steps)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 tdLayer(nn.Conv2d(in_planes, self.expansion * planes,
#                                   kernel_size=1, stride=stride, bias=False), self.nb_steps),
#                 warpBN(self.expansion * planes, bn_type, self.nb_steps)
#             )
#         self.spike3 = LIFLayer(**kwargs_spikes)
#
#     def forward(self, x):
#         out = self.spike1(self.bn1(self.conv1(x)))
#         out = self.spike2(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = self.spike3(out)
#         return out
#
#
# class ResNet19(nn.Module):
#     def __init__(self, block, num_block_layers, num_classes=10, in_channel=3, bn_type='', **kwargs_spikes):
#         super(ResNet19, self).__init__()
#         self.in_planes = 128
#         self.bn_type = bn_type
#         self.kwargs_spikes = kwargs_spikes
#         self.nb_steps = kwargs_spikes['nb_steps']
#         self.conv0 = nn.Sequential(
#             tdLayer(nn.Conv2d(in_channel, self.in_planes, kernel_size=3, padding=1, stride=1, bias=False),
#                     nb_steps=self.nb_steps),
#             warpBN(self.in_planes, bn_type, self.nb_steps),
#             LIFLayer(**kwargs_spikes)
#         )
#         self.layer1 = self._make_layer(block, 128, num_block_layers[0], stride=1)
#         self.layer2 = self._make_layer(block, 256, num_block_layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 512, num_block_layers[2], stride=2)
#         self.avg_pool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)), nb_steps=self.nb_steps)
#         self.classifier = nn.Sequential(
#             tdLayer(nn.Linear(512 * block.expansion, 256, bias=False), nb_steps=self.nb_steps),
#             LIFLayer(**kwargs_spikes),
#             tdLayer(nn.Linear(256, num_classes, bias=False), nb_steps=self.nb_steps),
#             Readout()
#         )
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride, self.bn_type, **self.kwargs_spikes))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out, _ = torch.broadcast_tensors(x, torch.zeros((self.nb_steps,) + x.shape))
#         out = self.conv0(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.avg_pool(out)
#         out = out.view(out.shape[0], out.shape[1], -1)
#         out = self.classifier(out)
#         return out

def spiking_resnet18(neuron: callable = None, num_classes=10, neuron_dropout=0, **kwargs):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes, neuron_dropout, neuron=neuron, **kwargs)


def spiking_resnet34(neuron: callable = None, num_classes=10, neuron_dropout=0, **kwargs):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes, neuron_dropout, neuron=neuron, **kwargs)


def spiking_resnet50(neuron: callable = None, num_classes=10, neuron_dropout=0, **kwargs):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes, neuron_dropout, neuron=neuron, **kwargs)


def spiking_resnet101(neuron: callable = None, num_classes=10, neuron_dropout=0, **kwargs):
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], num_classes, neuron_dropout, neuron=neuron, **kwargs)


def spiking_resnet152(neuron: callable = None, num_classes=10, neuron_dropout=0, **kwargs):
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], num_classes, neuron_dropout, neuron=neuron, **kwargs)

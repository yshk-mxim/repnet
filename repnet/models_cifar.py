"""CIFAR model definitions: ResNet-18 and ResNet-18 + Bottleneck.

Both use 2D DCT initialization from repnet.init.dct_init_2d.
"""
import torch.nn as nn
import torch.nn.functional as F


class IntermediateBottleneck2D(nn.Module):
    """1x1 conv bottleneck for 2D feature maps."""
    def __init__(self, channels, bn_dim):
        super().__init__()
        self.compress = nn.Conv2d(channels, bn_dim, 1)
        self.bn = nn.BatchNorm2d(bn_dim)
        self.expand = nn.Conv2d(bn_dim, channels, 1)

    def forward(self, x):
        z = F.gelu(self.bn(self.compress(x)))
        return x + self.expand(z)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18CIFAR(nn.Module):
    """ResNet-18 for CIFAR (3x3 stem, no max pool)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


class ResNet18BottleneckCIFAR(nn.Module):
    """ResNet-18 + information bottleneck at each stage."""
    def __init__(self, num_classes=10, bn_dim=12):
        super().__init__()
        self.bn_dim = bn_dim
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.ibn1 = IntermediateBottleneck2D(64, bn_dim)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.ibn2 = IntermediateBottleneck2D(128, bn_dim)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.ibn3 = IntermediateBottleneck2D(256, bn_dim)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.ibn4 = IntermediateBottleneck2D(512, bn_dim)
        self.bn_linear = nn.Linear(512, bn_dim)
        self.bn_act = nn.GELU()
        self.fc = nn.Linear(bn_dim, num_classes)

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.ibn1(self.layer1(out))
        out = self.ibn2(self.layer2(out))
        out = self.ibn3(self.layer3(out))
        out = self.ibn4(self.layer4(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        h = self.bn_act(self.bn_linear(out))
        return self.fc(h), h

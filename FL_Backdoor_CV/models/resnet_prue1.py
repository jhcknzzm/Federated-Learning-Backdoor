'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, cfg, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # print('cfg',cfg)
        self.cfg = cfg
        self.conv1 = nn.Conv2d(in_planes, cfg, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.conv2 = nn.Conv2d(cfg, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.size(), self.cfg)
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))


        out += self.shortcut(x)
        out = F.relu(out)
        return out

def downsample_basic_block(x, planes):
    x = nn.AvgPool2d(2,2)(x)
    zero_pads = torch.Tensor(
        x.size(0), planes - x.size(1), x.size(2), x.size(3)).zero_()
    if isinstance(x.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([x.data, zero_pads], dim=1))

    return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, cfg, num_classes=10):
        super(ResNet, self).__init__()
        n = 2
        self.in_planes = 64
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], cfg=cfg[0:n], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], cfg=cfg[n:2*n], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], cfg=cfg[2*n:3*n], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], cfg=cfg[3*n:4*n],  stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    # def _make_layer(self, block, planes, num_blocks, cfg, stride):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
    #     return nn.Sequential(*layers)
    def _make_layer(self, block, planes, num_blocks, cfg, stride=1):
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=planes*block.expansion)
        strides = [stride] + [1]*(num_blocks-1)

        layers = []
        layers.append(block(self.in_planes, planes, cfg[0], stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, cfg[i]))

        return nn.Sequential(*layers)

    # def _make_layer(self, block, planes, num_blocks, cfg, stride):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #     k = 0
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, cfg[k+1], stride))
    #         self.in_planes = planes * block.expansion
    #         k += 1
    #     return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.size())
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_prue(cfg):
    return ResNet(BasicBlock, [2,2,2,2], cfg)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()

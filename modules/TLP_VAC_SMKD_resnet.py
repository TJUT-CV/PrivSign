import torch
import torch.nn as nn
import torchvision.models as models
from .gabor_layer import GaborLayerLearnable
import torch.nn.functional as F
from torchjpeg import dct
# Define resnet BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

#
class ResNet18(nn.Module):
    def __init__(self, num_classes=1296,kernels2=1):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        self.gabor_layer = GaborLayerLearnable
        self.gabor_layer_conv = self.gabor_layer(in_channels=64, out_channels=64,
                                                 kernel_size=3, stride=1, padding=1, kernels=kernels2)
        # self.gabor_layer_conv_2 = self.gabor_layer(in_channels=64, out_channels=64,
        #                                          kernel_size=3, stride=1, padding=1, kernels=kernels2)
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        x_1 = self.gabor_layer_conv(x)
        x = x_1 + x
        x = self.layer1(x)
        # x_2 = self.gabor_layer_conv_2(x)
        # x = x_2 + x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18():
    model = ResNet18()
    # pretrained_resnet18 = models.resnet18(pretrained=True)
    # pretrained_state_dict = pretrained_resnet18.state_dict()

    #
    # custom_model_state_dict = model.state_dict()

    #
    # for name, param in custom_model_state_dict.items():
    #     if name in pretrained_state_dict:
    #         custom_model_state_dict[name].copy_(pretrained_state_dict[name])
    # model.load_state_dict(custom_model_state_dict)
    # for name, param in model.named_parameters():
    #     print(name, param.data)
    return model
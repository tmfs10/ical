from torch import nn as nn, Tensor
from torch.nn import functional as F

# from torchvision import models

import mc_dropout


class BayesianNet(mc_dropout.BayesianModule):
    def __init__(self, num_classes):
        super().__init__(num_classes)

        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,padding=1)
        self.conv1_drop = mc_dropout.MCDropout2d()
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,padding=1)
        self.conv2_drop = mc_dropout.MCDropout2d()
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,padding=1)
        self.conv3_drop = mc_dropout.MCDropout2d()
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3,padding=1)
        self.conv4_drop = mc_dropout.MCDropout2d()
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3,padding=1)
        self.conv5_drop = mc_dropout.MCDropout2d()
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        self.conv6_drop = mc_dropout.MCDropout2d()
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        self.conv7_drop = mc_dropout.MCDropout2d()
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        self.conv8_drop = mc_dropout.MCDropout2d()
        self.bn8 = nn.BatchNorm2d(256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.fc1 = nn.Linear(256, 512)
        self.fc1_drop = mc_dropout.MCDropout()
        self.fc2 = nn.Linear(512, 512)
        self.fc2_drop = mc_dropout.MCDropout()
        self.fc3 = nn.Linear(512, num_classes)

    def mc_forward_impl(self, input: Tensor):
        input = F.max_pool2d(F.relu(self.bn1(self.conv1(input))), kernel_size=2, stride=2)
        input = F.max_pool2d(F.relu(self.bn2(self.conv2(input))), kernel_size=2, stride=2)
        input = F.relu(self.bn3(self.conv3(input)))
        #input = self.conv4_drop(F.max_pool2d(F.relu(self.bn(self.conv4(input))), kernel_size=2, stride=2))
        input = F.max_pool2d(F.relu(self.bn4(self.conv4(input))), kernel_size=2, stride=2)
        input = F.relu(self.bn5(self.conv5(input)))
        #input = self.conv6_drop(F.max_pool2d(F.relu(self.conv6(input))), kernel_size=2, stride=2))
        input = F.max_pool2d(F.relu(self.bn6(self.conv6(input))), kernel_size=2, stride=2)
        input = F.relu(self.bn7(self.conv7(input)))
        #input = self.conv8_drop(F.max_pool2d(F.relu(self.bn(self.conv8(input))), kernel_size=2, stride=2))
        input = F.max_pool2d(F.relu(self.bn8(self.conv8(input))), kernel_size=2, stride=2)
        input = self.avgpool(input)
        #input = torch.flatten(input, 1)
        input = input.view(-1, 512)# * 7 * 7)
#        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc1_drop(F.relu(self.fc1(input)))
        input = self.fc2_drop(F.relu(self.fc2(input)))
        input = self.fc3(input)
        input = F.log_softmax(input, dim=1)
        return input


# class LambdaModule(nn.Module):
#     def __init__(self, func):
#         super().__init__()
#         self.func = func
#
#     def forward(self, *inputs):
#         return self.func(*inputs)
#
#
# class BayesianNet(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#
#         self.resnet = models.resnet18(pretrained=False,
#                                       num_classes=num_classes)
#         # Adapted resnet from:
#         # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
#         #LambdaModule(lambda x: x.expand((-1, 64, 28, 28))) #
#         self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         #self.resnet.layer1 = LambdaModule(lambda x: x.repeat((x.shape[0], 128, 28, 28)))
#         self.resnet.maxpool = LambdaModule(lambda x: x)
#
#     def forward(self, mc_input):
#         x, n = mc_dropout.mc_flatten(mc_input)
#         x = self.resnet(x)
#         x = F.log_softmax(x, dim=1)
#
#         return mc_dropout.mc_unflatten_B_K_(x, n)

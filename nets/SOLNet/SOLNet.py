import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
sys.path.append('.')
from torch.nn import Parameter
from nets.SOLNet.Conv_component import RepVGG
from nets.SOLNet.attention_component import DEAM, LGA
from thop import profile


# RepVGG-A1-my-self
class SOLNet(nn.Module):
    def __init__(self, deploy=False, use_checkpoint=False):
        super(SOLNet, self).__init__()

        width_multiplier = [0.5, 0.75, 0.75, 0.75, 1]
        self.backbone = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000, width_multiplier=width_multiplier, override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)
        # self.K1 = LGA(channels=int(64 * width_multiplier[0]), k_size=3, reduction=16, groups=4)
        # self.K2 = LGA(channels=int(64 * width_multiplier[1]), k_size=3, reduction=16, groups=4)
        self.QK = DEAM(int(512 * width_multiplier[4]), int(512 * width_multiplier[4]), 1, 3, 1, 1)
        self.PS = nn.PixelShuffle(2)

        # self.Conv1 = nn.Conv2d(int(512 * width_multiplier[4]), int(512 * width_multiplier[4]) // 4, 3, 1, 1)
        self.Conv1 = nn.Conv2d(int(512 * width_multiplier[4]) // 4, int(512 * width_multiplier[4]) // 4, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(int(512 * width_multiplier[4]) // 4)

        self.Conv2 = nn.Conv2d(int(256 * width_multiplier[3])+ int(512 * width_multiplier[4]) // 4, int(128 * width_multiplier[2]), 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(int(128 * width_multiplier[2]))
        self.feature_1 = nn.Conv2d(int(128 * width_multiplier[2]), 1, 1, 1, 0)

        self.Conv3 = nn.Conv2d(int(128 * width_multiplier[2]), int(64 * width_multiplier[1]), 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(int(64 * width_multiplier[1]))
        self.feature_2 = nn.Conv2d(int(64 * width_multiplier[1]), 1, 1, 1, 0)

        self.Conv4 = nn.Conv2d(int(64 * width_multiplier[1]), int(64 * width_multiplier[0]), 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(int(64 * width_multiplier[0]))
        self.feature_3 = nn.Conv2d(int(64 * width_multiplier[0]), 1, 1, 1, 0)

        self.Conv5 = nn.Conv2d(int(64 * width_multiplier[0]), int(64 * width_multiplier[0]),3, 1, 1)
        self.bn5 = nn.BatchNorm2d(int(64 * width_multiplier[0]))
        self.feature_4 = nn.Conv2d(int(64 * width_multiplier[0]), 1, 1, 1, 0)

        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

        self.Up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.Up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.Up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)


        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        layers_result = []
        x, layers_result = self.backbone(x)
        x = self.QK(x)
        # x = self.Conv1(x)
        x = self.PS(x)
        x = self.relu(self.bn1(self.Conv1(x)))

        x = torch.cat([x, layers_result[20]], dim=1)
        x = self.relu(self.bn2(self.Conv2(x)))


        feature_1 = self.feature_1(self.drop(x))

        x = self.Up_2(x)
        # x = torch.cat([x, layers_result[6]], dim=1)
        x = x + layers_result[6]
        x = self.relu(self.bn3(self.Conv3(x)))

        feature_2 = self.feature_2(self.drop(x))

        x = self.Up_2(x)
        # x = torch.cat([x, self.K2(layers_result[2])], dim=1)
        x = x + layers_result[2]
        x = self.relu(self.bn4(self.Conv4(x)))

        feature_3 = self.feature_3(self.drop(x))

        x = self.Up_2(x)
        # x = torch.cat([x, self.K1(layers_result[0])], dim=1)
        x = x + layers_result[0]
        x = self.relu(self.bn5(self.Conv5(x)))

        x = self.feature_4(self.drop(x))

        feature_1 = self.Up_8(feature_1)
        feature_2 = self.Up_4(feature_2)
        feature_3 = self.Up_2(feature_3)

        return feature_1, feature_2, feature_3, x, self.sigmoid(feature_1), self.sigmoid(feature_2), self.sigmoid(feature_3), self.sigmoid(x)


if __name__=='__main__':
    model = SOLNet(deploy=True)
    input = torch.randn(1, 3, 256, 256)
    flops, params = profile(model, inputs=(input,), verbose=False)
    print(f"Ours---->>>>>\nFLOPs: {flops / (10 ** 9)}G\nParams: {params / (10 ** 6)}M")

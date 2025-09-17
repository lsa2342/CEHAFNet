import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, kernel_num=4):
        super(Attention, self).__init__()
        self.in_channel = in_planes
        self.out_channel = out_planes
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.conv1d_1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False) 
        self.relu = nn.ReLU(inplace=True)
        self.out_channel_conv2d = nn.Conv2d(self.in_channel, self.out_channel, 1, bias=False)
        self.kernel_conv2d = nn.Conv2d(self.in_channel, kernel_num, 1, bias=False)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)

        x_inchannel = x.squeeze(-1).permute(0, 2, 1)
        x_inchannel = self.conv1d_1(x_inchannel)
        x_inchannel = x_inchannel.permute(0, 2, 1).unsqueeze(-1)
        x_inchannel = torch.sigmoid(x_inchannel)

        x_outchannel = x.squeeze(-1).permute(0, 2, 1)
        x_outchannel = self.conv1d_1(x_outchannel)
        x_outchannel = x_outchannel.permute(0, 2, 1).unsqueeze(-1)
        x_outchannel = self.out_channel_conv2d(x_outchannel)
        x_outchannel = torch.sigmoid(x_outchannel)

        x_kernel = x.squeeze(-1).permute(0, 2, 1)
        x_kernel = self.conv1d_1(x_kernel)
        x_kernel = x_kernel.permute(0, 2, 1).unsqueeze(-1)
        x_kernel = self.kernel_conv2d(x_kernel)
        x_kernel = x_kernel.view(x_kernel.size(0), -1, 1, 1, 1, 1)
        x_kernel = F.softmax(x_kernel, dim=1)

        return x_inchannel, x_outchannel, x_kernel

class EDE(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_num=1, kernel_size=3, stride=1, padding=0, groups=1, dilation=1):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        in_channel_attention, out_channel_attention, kernel_attention = self.attention(x)
        b, c, h, w = x.size()
        x = x * in_channel_attention
        x = x.reshape(1, -1, h, w)
        aggregate_weight = kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view([-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])

        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * b)
        output = output.view(b, self.out_planes, output.size(-2), output.size(-1))
        output = output * out_channel_attention
        return output

class LGA(nn.Module):
    def __init__(self, channels, k_size, reduction, groups):
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.avg_pool_1 = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2) 
        self.sigmoid = nn.Sigmoid()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool_2 = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels // 2, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels // 2, 1, bias=False)
        )

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, inputs):
        input_1, input_2 = inputs.chunk(2, dim=1)

        x = self.avg_pool_1(input_1)
        x = x.squeeze(-1).permute(0, 2, 1)
        x = self.conv(x)
        x = self.sigmoid(x)
        x = x.view(-1, self.channels // 2, 1, 1)
        x = input_1 * x.expand_as(input_1)

        max_pool = self.mlp(self.max_pool(input_2))
        avg_pool = self.mlp(self.avg_pool_2(input_2))
        channel_out = self.sigmoid(max_pool + avg_pool)
        y = channel_out * input_2
        # y = y.view(-1, self.input_channels // 2, 1, 1)

        x = torch.cat([x, y], dim=1)
        x = self.channel_shuffle(x, self.groups)
        return x


class PS(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super().__init__()
        self.hidden_channels = in_channels // (scale * scale)
        self.out_channels = out_channels
        self.ps = nn.PixelShuffle(scale)

        # self.conv = Conv(self.hidden_channels, out_channels, ksize, 1)

    def forward(self, x):
        x = self.ps(x)
        # x = self.conv(x)
        _, _, h, w = x.data.size()
        x = x.view(-1, self.out_channels, h, w)
        return x

class DEAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_num, k_size, stride, padding, reduction=16, groups=4):
        super().__init__()

        self.groups = groups
        self.K = LGA(channels=in_channels, k_size=3, reduction=reduction, groups=groups)
        self.enhance_conv = EDE(in_channels, in_channels, kernel_num, k_size, 
                                               stride, padding, groups=1, dilation=1)

        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(-1, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, num_channels, height, width)
        return x

    def forward(self, x):
        K = self.K(x)
        Q = self.enhance_conv(x)
        x = Q + K
        x = self.bn(x)
        x = self.relu(x)
        return x

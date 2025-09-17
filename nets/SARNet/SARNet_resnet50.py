from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format
from backbone.resnet import Backbone_ResNet50_in3, resnet50_rmfc


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class ModuleHelper:
    @staticmethod
    def BNReLU(num_features, inplace=True):
        return nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=inplace)
        )

    @staticmethod
    def BatchNorm2d(num_features):
        return nn.BatchNorm2d(num_features)

    @staticmethod
    def Conv3x3_BNReLU(in_channels, out_channels, stride=1, dilation=1, groups=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                      groups=groups, bias=False),
            ModuleHelper.BNReLU(out_channels)
        )

    @staticmethod
    def Conv1x1_BNReLU(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            ModuleHelper.BNReLU(out_channels)
        )

    @staticmethod
    def Conv1x1(in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)


class Conv3x3(nn.Module):
    def __init__(self, in_chs, out_chs, dilation=1, dropout=None):
        super(Conv3x3, self).__init__()

        if dropout is None:
            self.conv = ModuleHelper.Conv3x3_BNReLU(in_channels=in_chs, out_channels=out_chs, dilation=dilation)
        else:
            self.conv = nn.Sequential(
                ModuleHelper.Conv3x3_BNReLU(in_channels=in_chs, out_channels=out_chs, dilation=dilation),
                nn.Dropout(dropout)
            )

        initialize_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


class Conv1x1(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(Conv1x1, self).__init__()

        self.conv = ModuleHelper.Conv1x1_BNReLU(in_chs, out_chs)

        initialize_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_chs, out_chs, dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.doubel_conv = nn.Sequential(
            Conv3x3(in_chs=in_chs, out_chs=out_chs, dropout=dropout),
            Conv3x3(in_chs=out_chs, out_chs=out_chs, dropout=dropout)
        )

        initialize_weights(self.doubel_conv)

    def forward(self, x):
        out = self.doubel_conv(x)
        return out


class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(Classifier, self).__init__()

        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        initialize_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        y = self.sigmoid(out) * x
        return y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


def Upsample(x, size):
    return nn.functional.interpolate(x, size, mode='bilinear', align_corners=True)


class GateUnit(nn.Module):
    def __init__(self, in_chs):
        super(GateUnit, self).__init__()

        self.conv = nn.Conv2d(in_chs, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        initialize_weights(self.conv)

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)

        return y


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        initialize_weights(self.fc1, self.fc2)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out) * x
        return out


class Aux_Module(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(Aux_Module, self).__init__()

        self.aux = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        initialize_weights(self.aux)

    def forward(self, x):
        res = self.aux(x)
        return res


class SpatialGatherModule(nn.Module):
    """
        根据初始聚合上下文特征
        预测概率分布。
        使用软加权方法聚合上下文。
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size()
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)  # batch x k x c
        return ocr_context


class ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''

    def __init__(self, in_channels, key_channels, scale=1, bn_type=None):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        return context


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1, bn_type=None):
        super(SpatialOCR_Module, self).__init__()

        self.object_context_block = ObjectAttentionBlock(in_channels, key_channels, scale, bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SARNet_r(nn.Module):
    def __init__(self, channel=32):
        super(SARNet_r, self).__init__()
        # ---- ResNet50 Backbone ----
        (
            self.encoder1,
            self.encoder2,
            self.encoder4,
            self.encoder8,
            self.encoder16,
        ) = Backbone_ResNet50_in3()

        # Lateral layers
        reduction_dim = 256
        self.lateral_conv0 = Conv3x3(64, 64)
        self.lateral_conv1 = Conv3x3(256, reduction_dim)
        self.lateral_conv2 = Conv3x3(512, reduction_dim, dropout=0.1)
        self.lateral_conv3 = Conv3x3(1024, reduction_dim, dropout=0.1)
        self.lateral_conv4 = Conv3x3(2048, reduction_dim, dropout=0.1)

        # OCR
        self.conv3x3_ocr = ModuleHelper.Conv3x3_BNReLU(768, 256)
        self.ocr_aux = nn.Sequential(
            ModuleHelper.Conv1x1_BNReLU(reduction_dim * 3, reduction_dim),
            Classifier(reduction_dim, num_classes=1)
        )
        self.cls = Classifier(reduction_dim, num_classes=1)

        self.ocr_gather_head = SpatialGatherModule(cls_num=1)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=256, key_channels=256, out_channels=256, scale=1,
                                                 dropout=0.05)

        # 注意力门控模型5-->4
        self.CAM_5_0 = ChannelAttention(in_planes=reduction_dim)
        self.CAM_5_1 = ChannelAttention(in_planes=reduction_dim)
        self.SAM_5 = SpatialAttention()
        # self.SAM = SpatialAttentionModule()

        self.gamma_5 = nn.Parameter(torch.ones(1))
        self.gate_5 = nn.Sequential(
            nn.Conv2d(256 + 256, 1, 1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.conv_out_5 = DecoderBlock(2, 1, dropout=0.0)

        # 注意力门控模型4-->3
        self.CAM_4_0 = ChannelAttention(in_planes=reduction_dim)
        self.CAM_4_1 = ChannelAttention(in_planes=reduction_dim)
        self.SAM_4 = SpatialAttention()
        self.gamma_4 = nn.Parameter(torch.ones(1))
        self.gate_4 = nn.Sequential(
            nn.Conv2d(256 + 256, 1, 1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.conv_out_4 = DecoderBlock(2, 1, dropout=0.0)

        # 注意力门控模型3-->2
        self.CAM_3_0 = ChannelAttention(in_planes=reduction_dim)
        self.CAM_3_1 = ChannelAttention(in_planes=reduction_dim)
        self.SAM_3 = SpatialAttention()
        self.gamma_3 = nn.Parameter(torch.ones(1))
        self.gate_3 = nn.Sequential(
            nn.Conv2d(256 + 256, 1, 1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.conv_out_3 = DecoderBlock(2, 1, dropout=0.0)

        # 注意力门控模型2-->1
        self.CAM_2_0 = ChannelAttention(in_planes=reduction_dim)
        self.CAM_2_1 = ChannelAttention(in_planes=reduction_dim)
        self.SAM_2 = SpatialAttention()
        self.gamma_2 = nn.Parameter(torch.ones(1))
        self.gate_2 = nn.Sequential(
            nn.Conv2d(256 + 256, 1, 1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.conv_out_3 = DecoderBlock(2, 1, dropout=0.0)

        # 注意力门控模型1-->0
        self.CAM_1_0 = ChannelAttention(in_planes=reduction_dim)
        self.CAM_1_1 = ChannelAttention(in_planes=64)
        self.SAM_1 = SpatialAttention()
        self.gamma_1 = nn.Parameter(torch.ones(1))
        self.gate_1 = nn.Sequential(
            nn.Conv2d(256 + 64, 1, 1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.conv_out_3 = DecoderBlock(2, 1, dropout=0.0)

        self.ra4_conv1 = BasicConv2d(256, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)

        self.ra3_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.ra2_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.ra1_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.ra1_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra1_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra1_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.ra0_conv1 = BasicConv2d(64, 64, kernel_size=1)
        self.ra0_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra0_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra0_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x0 = self.encoder1(x)
        x1 = self.encoder2(x0)
        x2 = self.encoder4(x1)
        x3 = self.encoder8(x2)
        x4 = self.encoder16(x3)

        lateral_x0 = self.lateral_conv0(x0)
        lateral_x1 = self.lateral_conv1(x1)
        lateral_x2 = self.lateral_conv2(x2)
        lateral_x3 = self.lateral_conv3(x3)
        lateral_x4 = self.lateral_conv4(x4)

        lateral_x3 = F.interpolate(lateral_x3, scale_factor=2, mode='bilinear')
        lateral_x4 = F.interpolate(lateral_x4, scale_factor=4, mode='bilinear')

        out_cat = torch.cat((lateral_x2, lateral_x3, lateral_x4), 1)

        out_aux = self.ocr_aux(out_cat)

        feats = self.conv3x3_ocr(out_cat)
        context = self.ocr_gather_head(feats, out_aux)
        g_feats = self.ocr_distri_head(feats, context)

        ra5_feat = self.cls(g_feats)

        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8,
                                      mode='bilinear')

        ### 5-->4 ###
        x = -1 * (torch.sigmoid(ra5_feat)) + 1
        g_feats_temp = g_feats.mul(self.CAM_5_0(g_feats))
        lateral_x4_temp = lateral_x4.mul(self.CAM_5_1(lateral_x4))
        out_cat_cam_5 = torch.cat((lateral_x4_temp, g_feats_temp), 1)
        lateral_gate_x4 = self.gamma_5 * self.gate_5(out_cat_cam_5) * lateral_x4_temp
        x = x.expand(-1, 256, -1, -1).mul(lateral_gate_x4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)

        refine_out_4 = ra4_feat + ra5_feat  # bs,1,44,44
        lateral_map_4 = F.interpolate(refine_out_4, scale_factor=8, mode='bilinear')

        ### 4-->3 ###
        x = -1 * (torch.sigmoid(refine_out_4)) + 1
        lateral_x4_temp = lateral_x4.mul(self.CAM_4_0(lateral_x4))
        lateral_x3_temp = lateral_x3.mul(self.CAM_4_1(lateral_x3))

        out_cat_cam_3 = torch.cat((lateral_x3, lateral_x4_temp), 1)
        lateral_gate_x3 = self.gamma_4 * self.gate_4(out_cat_cam_3) * lateral_x3_temp
        x = x.expand(-1, 256, -1, -1).mul(lateral_gate_x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)

        refine_out_3 = ra3_feat + refine_out_4
        lateral_map_3 = F.interpolate(refine_out_3, scale_factor=8, mode='bilinear')

        ### 3-->2 ###
        x = -1 * (torch.sigmoid(refine_out_3)) + 1
        lateral_x3_temp = lateral_x3.mul(self.CAM_3_0(lateral_x3))
        lateral_x2_temp = lateral_x2.mul(self.CAM_3_1(lateral_x2))
        out_cat_cam_2 = torch.cat((lateral_x2, lateral_x3_temp), 1)
        lateral_gate_x2 = self.gamma_3 * self.gate_3(out_cat_cam_2) * lateral_x2_temp
        x = x.expand(-1, 256, -1, -1).mul(lateral_gate_x2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        refine_out_2 = ra2_feat + refine_out_3
        lateral_map_2 = F.interpolate(refine_out_2, scale_factor=8, mode='bilinear')

        ### 2-->1 ###
        refine_crop_1 = F.interpolate(refine_out_2, scale_factor=2, mode='bilinear')
        x2_crop_1 = F.interpolate(lateral_x2, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(refine_crop_1)) + 1
        lateral_x2_temp = x2_crop_1.mul(self.CAM_2_0(x2_crop_1))
        lateral_x1_temp = lateral_x1.mul(self.CAM_2_1(lateral_x1))
        out_cat_cam_1 = torch.cat((lateral_x2_temp, lateral_x1_temp), 1)
        lateral_gate_x1 = self.gamma_2 * self.gate_2(out_cat_cam_1) * lateral_x1_temp

        x = x.expand(-1, 256, -1, -1).mul(lateral_gate_x1)
        x = self.ra1_conv1(x)
        x = F.relu(self.ra1_conv2(x))
        x = F.relu(self.ra1_conv3(x))
        ra1_feat = self.ra1_conv4(x)
        refine_out_1 = ra1_feat + refine_crop_1
        lateral_map_1 = F.interpolate(refine_out_1, scale_factor=4, mode='bilinear')

        ### 1-->0 ###
        refine_crop_0 = F.interpolate(refine_out_1, scale_factor=2, mode='bilinear')
        x1_crop_1 = F.interpolate(lateral_x1, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(refine_crop_0)) + 1
        lateral_x1_temp = x1_crop_1.mul(self.CAM_1_0(x1_crop_1))
        lateral_x0_temp = lateral_x0.mul(self.CAM_1_1(lateral_x0))
        out_cat_cam_0 = torch.cat((lateral_x1_temp, lateral_x0_temp), 1)
        lateral_gate_x0 = self.gamma_1 * self.gate_1(out_cat_cam_0) * lateral_x0_temp

        x = x.expand(-1, 64, -1, -1).mul(lateral_gate_x0)
        x = self.ra0_conv1(x)
        x = F.relu(self.ra0_conv2(x))
        x = F.relu(self.ra0_conv3(x))
        ra0_feat = self.ra0_conv4(x)
        refine_out_0 = ra0_feat + refine_crop_0
        lateral_map_0 = F.interpolate(refine_out_0, scale_factor=2, mode='bilinear')
        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1, lateral_map_0


if __name__ == '__main__':
    from torchvision.transforms import v2
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pil2tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    tensor2pil = v2.ToPILImage()
    preprocess = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((256, 256)),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_idx = '0388'  # 0257 0092 0388 0354 0384
    image_path = f'../data/ORSSD/test/image/{img_idx}.jpg'
    gt_path = f'../data/ORSSD/test/gt/{img_idx}.png'
    edge_path = f'../data/ORSSD/test/emfi_edge/{img_idx}.png'

    image = Image.open(image_path).convert('RGB')
    img = preprocess(image).to(device).unsqueeze(0)
    gt = pil2tensor(Image.open(gt_path).convert('L')).to(device).unsqueeze(0)
    edge = pil2tensor(Image.open(gt_path).convert('L')).to(device).unsqueeze(0)

    loss = 0.0
    model = SARNet_r().to(device)
    # weight = 'models/CEHAFNet_ORSSD/20250428_095305-73659e1c-5CEk5p2NoReLU-GuideNoBR+UpConv-NmadpPRI-0.1e-4-6/LowestLoss.pth'
    # model.load_state_dict(torch.load(weight, map_location='cuda'))
    model.eval()

    out = model(img)
    output = out[0]
    img2show = output.squeeze(0).cpu()
    output_img = tensor2pil(img2show)
    output_img.show()

    flops, param = profile(model, [img, ])
    flops, param = clever_format([flops, param], "%.2f")
    print("flops are :" + flops, "params are :" + param)

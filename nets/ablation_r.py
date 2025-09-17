# coding=<utf-8>
import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
from thop import clever_format, profile

from backbone import resnet, vgg
from backbone.pvtv2 import pvt_v2_b2

# from backbone import vgg
# from backbone import resnet
from utils.visualizer import get_local

project_root = '/home/lsa/Shared/ORSI-SOD'
os.chdir(project_root)


# import softpool_cuda
# from SoftPool import soft_pool2d, SoftPool2d

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x



class F(nn.Module):
    """ v3.1: 添加残差，保证不比解码之后的x_d效果差 """

    def __init__(self, in_channels, out_channels):
        super(F, self).__init__()

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)

    @get_local('x_e', 'x_d', 'x_mul', 'x_e_c', 'out')
    def forward(self, x_d, x_e):
        identity = x_d
        x_e_c = self.conv1(x_e)

        x_mul = x_d * x_e
        x_mul_c = self.conv2(x_mul)

        out = x_mul_c + x_e_c
        out = self.conv(out)
        # out = self.bn(out)
        out = out + identity
        # out = self.relu(out_axd)

        return out


class CAA(nn.Module):
    """
        CAAv2
            CRAv2.1: 不使用F.relu激活x_local => 效果不大
            CRAv2.2: 使用F.relu，AvgAP改为Norm   #TODO
            CRAv2.3: Conv3D + norm
        CAAv3: Relu(local_1ch) + norm + PReluCk1ch1-o
            v3.1: Relu(local_inchs) + norm + PReluCk1ch1-o
            v3.2: Relu(local_inchs) + PReluCk1ch1-o + norm / torch.sqrt(torch.tensor(x_local_sub.size(1)))
            v3.3: Relu(local_inchs) + PReluCk1chi-o + norm / torch.sqrt(torch.tensor(x_local_sub.size(1)))
            v3.3.1: Relu(local_inchs) + PReluCk1chi-o + AP1
        CAAv4:
            v4.1.1: cat[x-(local_AP), x-GAP(x)]
            v4.1.2: cat[PReluCk1i1[x-(local_AP)], PReluCk1i1[x-GAP(x)]]
                    norm / torch.sqrt(torch.tensor(x_local_sub.size(1))) * x
                    pred Ci1k7p3
            v4.1.3: Ci1k7p3
                v4.1.3.1: Ci1k7p3 self.sigmoid(contrast) * x
            v4.1.4: pred Ci1k1p0
            v4.2: cat[x_local-(local_AP), x-GAP(x)]
    """

    def __init__(self, in_channels):
        super(CAA, self).__init__()
        self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.adpavg = nn.AdaptiveAvgPool2d(1)
        self.contrast = nn.Conv2d(in_channels, 2, kernel_size=1, padding=0, bias=False)
        self.activate = nn.PReLU(num_parameters=2)
        self.conv_norm = nn.Conv2d(2, in_channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    @get_local('x_local_avg', 'x_local_sub', 'x_global_avg', 'x_global_sub',
               'contrast_l', 'contrast_g', 'contrast_l_norm', 'contrast_g_norm', 'contrast_norm', 'contrast_score')
    def forward(self, x):
        # Code will be released soon

        return contrast_score


class CEM(nn.Module):
    """
    =>  v2.1.0: x*cra + Ck3(cra)*x
        v2.1.1: Convk1(cat) fusion
        v2.1.2: Convk3(cat) fusion #相较于2.1.1提升不大
        v2.1.3: 加残差 #TODO
        v2.2: Gate 适应cat两个分支
        v2.4: BGate
        v2.5: TriBanch Gate: attention-C1-in
            v2.5.1: TriGateSoftmax
            v2.5.2: TriGateSoftmax Cin-CRA-Ck1(CRA)
            v2.5.3: Cin-CRA-Cinchsk7(CRA)
            v2.5.4: Cin-CRA-Cinch1k7(CRA)
        v3: CRA1+Ci1o1k7(norm(dim1))+x
            v3.1: Ck3 + CRA1*x + Ci1o1k7(norm(dim1))*x + x
            v3.2: CRA1*x + Ci1o1k7(norm(dim1))*x + x
    """

    def __init__(self, in_channels):
        super(CEM, self).__init__()
        # self.conv3_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv3_2 = nn.Conv2d(1, 1, kernel_size=7, padding=3)  # 5 2
        self.cra = CRA(in_channels)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
        #     nn.Sigmoid()
        # )

    @get_local('x1', 'x1_norm', 'Ck3cra', 'x2', 'out')
    def forward(self, x):
        id_x = x
        # id_x = self.conv3_1(x)

        cra = self.cra(x)
        x1 = x * cra

        x1_norm = torch.norm(x1, p=2, dim=1, keepdim=True)  #/ torch.sqrt(torch.tensor(x1.shape[1]))
        Ck3cra = self.conv3_2(x1_norm)
        x2 = Ck3cra * x1

        # concat = torch.cat([x1, x2], dim=1)
        # weights = self.conv(concat)
        # out = weights[:, 0:1] * x1 + weights[:, 1:2] * x2
        out = x1 + x2 + id_x
        return out


class AEDB(nn.Module):
    """
        v3.5.0: idK1-Cca-sa+id
            v3.5.1: idk1-Cca-Csa+id bias=FFFT
            v3.5.1.1: idk1-Cca-C(sa*semantic)+id bias=FFFT
            v3.5.2: idk1-Cca-Csa+id bias=FTTT
            v3.5.3: idk1-Cca(ratio=out_chs)-Csa+id bias=FTTT    # 0257中间相似部分去除，0388阴影不能区分
            v3.5.4: idk1-Cca(ratio=out_chs)-sa-C+idk3 bias=FTTT
            v3.5.5: Cca(i,ratio=out_chs)-sa*out0-C+idk1 bias=FTT
            v3.5.5.1: Cca(i,ratio=i_chs)-sa*semantic-C+idk3 bias=FTT
        =>    v3.5.5.2: Cca(o,ratio=o_chs)-sa*semantic-C+idk1 bias=FTT
            v3.5.5.2.1: Cca(o,ratio=o_chs)-sa*out0b-C+idk1 bias=FTT
            v3.5.5.3: ca(i,ratio=i_chs)C-sa*semantic-C+idk1 bias=FTT
            v3.5.6: Cii-eca-sa-Cio+idk3 bias=FTTT
        v3.6.0: 添加一个分支剔除冗余信息
            v3.6.1: sa(semantic) * out1 bias=FFFT =》效果不大
            v3.6.2: idk1-Cca(ratio=out_chs)+residual-sa(*semantic_r)-relu-C+id bias=FTTT
            v3.6.2.1: idk1-Cca(ratio=out_chs)+residual-sa(*semantic)-relu-C+id bias=FTTT
            v3.6.3: idk1-Cca(ratio=out_chs)+residual-sa(*semantic_r)-C+id bias=FTTT
        v3.7.4.2： Ck131 bias=TTT - CA ratio * 2
        v3.7.4.3： CA ratio
            v3.7.4.3.1： CA ratio out1 = out0 * (1 + ca_w + sa_w * ca_w)
    """

    def __init__(self, in_channels, out_channels):
        super(AEDB, self).__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, padding=0)
        self.ca = ChannelAttention(out_channels, 1)
        self.sa = SpatialAttention()

    @get_local('out0', 'ca_w', 'sa_w', 'out1', 'out2', 'identity', 'out3')
    def forward(self, x):
        out0 = self.relu(self.bn1(self.conv1(x)))
        ###### 

        out2 = self.relu(self.bn2(self.conv2(out1)))

        identity = self.shortcut(x)
        out3 = self.conv(torch.cat([out2, identity], dim=1))
        return out3


class Up(nn.Module):
    """
    v1: TConv+Conv3
    v2: TConv+Conv3(UpSample)
        v2.1 [TConv+Conv3(UpSample)]BN
        v2.2 [TConv+Conv3(UpSample)]BN
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.Up1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # self.bn1 = nn.BatchNorm2d(out_channels)

        self.Up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_channels)

        # self.relu = nn.ReLU(inplace=True)

    @get_local('up2')
    def forward(self, x):
        # up1 = self.Up1(x)
        up2 = self.conv(self.Up2(x))
        # up_out = up1 + up2
        return up2


class Decoder(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, decode, final=False):
        super(Decoder, self).__init__()

        self.final = final
        self.concat = F(in_channels + in_channels, in_channels) if not self.final else None
        self.decode = decode(in_channels, out_channels)

        self.Up = Up(out_channels, out_channels)

    def forward(self, x_d, x_e=None):
        if x_e is not None and self.concat:
            x_d = self.concat(x_e, x_d)

        x_d = self.decode(x_d)
        x_d = self.Up(x_d)

        return x_d


class SalHead(nn.Module):
    """
    v0.1: Upsample + Convk3
    """

    def __init__(self, in_channels, scale_factor=1):
        super(SalHead, self).__init__()
        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.upsample(x)
        out = self.conv(x)
        return out


class HAFM(nn.Module):
    """
        v0.1.2: stack-Ck1b=F+ch_Att(eca)   Sigmoid()
        =>    v0.1.2.1: stack-Ck1b=F+ch_Att(eca)   *替换cat img_cat
            v0.1.2.2: stack-Ck1b=F+ch_Att(eca)   *替换cat img_s
        v0.1.3: stack-Ck1b=F+ch_Att(eca)   Sigmoid() Conv1d self.num_levels-chs
            v0.1.3.1: stack-Ck1b=T+ch_Att(eca)   Sigmoid() Conv1d self.num_levels-chs
        v0.1.4: stack-Ck1b=F+ch_Att(eca) * img_s  Sigmoid()
    """

    def __init__(self, num_levels):
        super(HAFM, self).__init__()
        self.num_levels = num_levels
        self.conv = nn.Conv2d(self.num_levels, self.num_levels, kernel_size=1, padding=0, bias=False)
        self.prelu = nn.PReLU(num_parameters=self.num_levels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.pre = nn.Conv2d(self.num_levels, 1, kernel_size=3, padding=1)
        self.score = nn.Softmax(dim=1)
        # self.score = nn.Sigmoid()

    @get_local('img_s', 'l_avg', 'l_w', 'fus')
    def forward(self, *img):
        # img_s = torch.cat(img, dim=1)
        img_s = torch.stack(img, 1).squeeze(2)
        img_cat = self.prelu(self.conv(img_s))  # [1, num, H, W] torch.stack(img, 1).squeeze(2)
        l_avg = self.avg_pool(img_cat)
        ### 
        levels_weight = self.score(l_w)
        fus = img_s * levels_weight
        salmap = self.pre(fus)

        return salmap


class CEHAFNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, block=AEDB):
        super(CEHAFNet, self).__init__()
        # vgg:      64, 128, 256, 512, 512
        # resnet:   64, 256, 512, 1024, 2048
        # pvt_v2:   64, 128, 320, 512   64, 256, 512, 1024
        backbone_out_chs = [64, 256, 512, 1024, 2048]
        param_channels = [64, 128, 256, 512, 1024]

        # backbone = models.resnet50(pretrained=False)
        self.feature1, self.feature2, self.feature3, self.feature4, self.feature5 = resnet.resnet50_rmfc()
        # state_dict = torch.load('./backbone/pretrained/resnet50-19c8e357.pth', weights_only=True)
        # backbone.load_state_dict(state_dict)

        # backbone = res2net50_v1b_26w_4s(pretrained=True)

        # self.backbone = pvt_v2_b2()
        # path = 'backbone/pretrained/pvt_v2_b2.pth'
        # save_model = torch.load(path)
        # model_dict = self.backbone.state_dict()
        # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # model_dict.update(state_dict)
        # self.backbone.load_state_dict(model_dict)

        # self.feature1 = vgg.vgg16().conv1 #self.backbone[0] # nn.Sequential(*list(backbone.children())[:3])  #vgg.vgg16().conv1    # 64 channels
        # self.feature2 = vgg.vgg16().conv2 #self.backbone[1] # nn.Sequential(*list(backbone.children())[3:5]) #vgg.vgg16().conv2    # 256 channels
        # self.feature3 = vgg.vgg16().conv3 #self.backbone[2] # backbone.layer2                                #vgg.vgg16().conv3    # 512 channels
        # self.feature4 = vgg.vgg16().conv4 #self.backbone[3] # backbone.layer3                                #vgg.vgg16().conv4    # 1024 channels
        # self.feature5 = vgg.vgg16().conv5 #self.backbone[4]                                                  #vgg.vgg16().conv5    # 2048 channels

        self.encoder1 = BasicConv2d(backbone_out_chs[0], param_channels[0], kernel_size=3, padding=1)
        self.encoder2 = BasicConv2d(backbone_out_chs[1], param_channels[1], kernel_size=3, padding=1)
        self.encoder3 = BasicConv2d(backbone_out_chs[2], param_channels[2], kernel_size=3, padding=1)
        self.encoder4 = BasicConv2d(backbone_out_chs[3], param_channels[3], kernel_size=3, padding=1)
        self.encoder5 = BasicConv2d(backbone_out_chs[4], param_channels[4], kernel_size=3, padding=1)

        self.ce1 = CEM(param_channels[0])
        self.ce2 = CEM(param_channels[1])
        self.ce3 = CEM(param_channels[2])
        self.ce4 = CEM(param_channels[3])
        self.ce5 = CEM(param_channels[4])

        self.decoder5 = Decoder(param_channels[4], param_channels[3], decode=block, final=True)
        self.decoder4 = Decoder(param_channels[3], param_channels[2], decode=block)
        self.decoder3 = Decoder(param_channels[2], param_channels[1], decode=block)
        self.decoder2 = Decoder(param_channels[1], param_channels[0], decode=block)
        self.decoder1 = Decoder(param_channels[0], n_classes, decode=block)

        # Supervised
        self.s5 = SalHead(param_channels[3], 16)  # // 2
        self.s4 = SalHead(param_channels[2], 8)  # 16 // 2
        self.s3 = SalHead(param_channels[1], 4)  # 8 // 2
        self.s2 = SalHead(param_channels[0], 2)  # 4 // 2
        self.s1 = SalHead(n_classes)  # 2 , 0.5
        self.s = Fusion(5)
        # self.s = Fusion1([n_classes, param_channels[0], param_channels[1], param_channels[2], param_channels[3]])
        # self.s = Fusion2([n_classes, param_channels[0], param_channels[1], param_channels[2], param_channels[3]], 16)

        self._initialize_weights()  # 只在__init__()中初始化一次权值

    def forward(self, x):
        feat1 = self.feature1(x)
        feat2 = self.feature2(feat1)
        feat3 = self.feature3(feat2)
        feat4 = self.feature4(feat3)
        feat5 = self.feature5(feat4)
        # pvt = self.backbone(x)
        # feat1 = pvt[0]
        # feat2 = pvt[1]
        # feat3 = pvt[2]
        # feat4 = pvt[3]

        # Encoder
        x_e1 = self.encoder1(feat1)
        x_e2 = self.encoder2(feat2)
        x_e3 = self.encoder3(feat3)
        x_e4 = self.encoder4(feat4)
        x_e5 = self.encoder5(feat5)

        # Enhanced
        x_e1 = self.ce1(x_e1)
        x_e2 = self.ce2(x_e2)
        x_e3 = self.ce3(x_e3)
        x_e4 = self.ce4(x_e4)
        x_e5 = self.ce5(x_e5)

        # Decoder
        x_d5 = self.decoder5(x_e5)  # 1024 channels
        x_d4 = self.decoder4(x_d5, x_e4)
        x_d3 = self.decoder3(x_d4, x_e3)
        x_d2 = self.decoder2(x_d3, x_e2)
        x_d1 = self.decoder1(x_d2, x_e1)

        # supervised
        s5 = self.s5(x_d5)
        s4 = self.s4(x_d4)
        s3 = self.s3(x_d3)
        s2 = self.s2(x_d2)
        s1 = self.s1(x_d1)
        s = self.s(s5, s4, s3, s2, s1)
        # s = self.s(x_d1, x_d2, x_d3, x_d4, x_d5)

        return torch.sigmoid(s), torch.sigmoid(s2), torch.sigmoid(s2), torch.sigmoid(s3), torch.sigmoid(
            s4), torch.sigmoid(s5)
        # return s, s1, s2, s3, s4, s5

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ChannelAttention):
                nn.init.kaiming_normal_(m.fc1.weight, mode='fan_out', nonlinearity='relu')
                if m.fc1.bias is not None:
                    nn.init.constant_(m.fc1.bias, 0)
                nn.init.kaiming_normal_(m.fc2.weight, mode='fan_out', nonlinearity='relu')
                if m.fc2.bias is not None:
                    nn.init.constant_(m.fc2.bias, 0)
            elif isinstance(m, SpatialAttention):
                nn.init.kaiming_normal_(m.conv1.weight, mode='fan_out', nonlinearity='relu')
                if m.conv1.bias is not None:
                    nn.init.constant_(m.conv1.bias, 0)


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
    image_path = f'./data/ORSSD/test/image/{img_idx}.jpg'
    gt_path = f'./data/ORSSD/test/gt/{img_idx}.png'

    image = Image.open(image_path).convert('RGB')
    gt = pil2tensor(Image.open(gt_path)).squeeze()
    img = preprocess(image).to(device).unsqueeze(0)  # 因为在batchnorm会检查类型，必须是4D

    model = CEHAFNet(3, 1).to(device)
    # weight = 'models/CEHAFNet_ORSSD/20250428_095305-73659e1c-5CEk5p2NoReLU-GuideNoBR+UpConv-NmadpPRI-0.1e-4-6/LowestLoss.pth'
    # model.load_state_dict(torch.load(weight, map_location='cuda'))
    model.eval()

    flops, param = profile(model, [img, ])
    flops, param = clever_format([flops, param], "%.2f")
    print("flops are :" + flops, "params are :" + param)

    out = model(img)
    output = out[0]
    img2show = output.squeeze(0).to(device)
    output_img = tensor2pil(img2show)
    output_img.show()

    tb_dir = '../tensorboard/temp'
    Path(tb_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=f'{tb_dir}')
    writer.add_graph(model, img)
    writer.close()

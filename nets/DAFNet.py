import os
from pathlib import Path

import numpy as np
from PIL import Image
from thop import profile, clever_format
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torchvision import models, transforms
import warnings

warnings.filterwarnings('ignore')


def align_number(number, N):
    assert type(number) == int
    num_str = str(number)
    assert len(num_str) <= N
    return (N - len(num_str)) * '0' + num_str


def unload(x):
    y = x.squeeze().cpu().data.numpy()
    return y


def min_max_normalization(x):
    x_normed = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x_normed


def convert2img(x):
    return Image.fromarray(x * 255).convert('L')


def save_smap(smap, path, negative_threshold=0.25):
    # smap: [1, H, W]
    if torch.max(smap) <= negative_threshold:
        smap[smap < negative_threshold] = 0
        smap = convert2img(unload(smap))
    else:
        smap = convert2img(min_max_normalization(unload(smap)))
    smap.save(path)


def cache_model(model, path, multi_gpu):
    if multi_gpu == True:
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def initiate(md, path):
    md.load_state_dict(torch.load(path))


def DS2(x):
    return F.avg_pool2d(x, 2)


def DS4(x):
    return F.avg_pool2d(x, 4)


def DS8(x):
    return F.avg_pool2d(x, 8)


def DS16(x):
    return F.avg_pool2d(x, 16)


def US2(x):
    return F.interpolate(x, scale_factor=2, mode='bilinear')


def US4(x):
    return F.interpolate(x, scale_factor=4, mode='bilinear')


def US8(x):
    return F.interpolate(x, scale_factor=8, mode='bilinear')


def US16(x):
    return F.interpolate(x, scale_factor=16, mode='bilinear')


def RC(F, A):
    return F * A + F


def clip(inputs, rho=1e-15, mu=1 - 1e-15):
    return inputs * (mu - rho) + rho


def BCELoss_OHEM(batch_size, pred, gt, num_keep):
    loss = torch.zeros(batch_size).cuda()
    for b in range(batch_size):
        loss[b] = F.binary_cross_entropy(pred[b, :, :, :], gt[b, :, :, :])
        sorted_loss, idx = torch.sort(loss, descending=True)
        keep_idx = idx[0:num_keep]
        ohem_loss = loss[keep_idx]
        ohem_loss = ohem_loss.sum() / num_keep
    return ohem_loss


def proc_loss(losses, num_total, prec=4):
    loss_for_print = []
    for l in losses:
        loss_for_print.append(np.around(l / num_total, prec))
    return loss_for_print


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, bias=False)  # infer a one-channel attention map

    def forward(self, ftr):
        # ftr: [B, C, H, W]
        ftr_avg = torch.mean(ftr, dim=1, keepdim=True)  # [B, 1, H, W], average
        ftr_max, _ = torch.max(ftr, dim=1, keepdim=True)  # [B, 1, H, W], max
        ftr_cat = torch.cat([ftr_avg, ftr_max], dim=1)  # [B, 2, H, W]
        att_map = F.sigmoid(self.conv(ftr_cat))  # [B, 1, H, W]
        return att_map


class CPA(nn.Module):
    # Cascaded Pyramid Attention
    def __init__(self, in_channels):
        super(CPA, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv_1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.conv_2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.SA0 = SpatialAttention()
        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()

    def forward(self, ftr):
        # ftr: [B, C, H, W]
        d0 = self.conv_0(ftr)  # [B, C/2, H, W]
        d1 = self.conv_1(DS2(ftr))  # [B, C/4, H/2, W/2]
        d2 = self.conv_2(DS4(ftr))  # [B, C/4, H/4, W/4]
        # level-2
        a2 = self.SA2(d2)  #  [B, 1, H/4, W/4]
        d2 = a2 * d2 + d2  # [B, C/4, H/4, W/4]
        # level-1
        d1 = torch.cat([d1, US2(d2)], dim=1)  # [B, C/2, H/2, W/2]
        a1 = self.SA1(d1)  # [B, 1, H/2, W/2]
        d1 = a1 * d1 + d1  # [B, C/2, H/2, W/2]
        # level-0
        d0 = torch.cat([d0, US2(d1)], dim=1)  # [B, C, H, W]
        a0 = self.SA0(d0)  # [B, 1, H, W]
        return a0, d0


class ChannelRecalibration(nn.Module):
    def __init__(self, in_channels):
        super(ChannelRecalibration, self).__init__()
        inter_channels = in_channels // 4  # channel squeezing
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_fc = nn.Sequential(nn.Linear(in_channels, inter_channels, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(inter_channels, in_channels, bias=False))
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_fc = nn.Sequential(nn.Linear(in_channels, inter_channels, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(inter_channels, in_channels, bias=False))

    def forward(self, ftr):
        # ftr: [B, C, H, W]
        ftr_avg = self.avg_fc(self.avg_pool(ftr).squeeze(-1).squeeze(-1))  # [B, C]
        ftr_max = self.max_fc(self.max_pool(ftr).squeeze(-1).squeeze(-1))  # [B, C]
        weights = F.sigmoid(ftr_avg + ftr_max).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        out = weights * ftr
        return out


class GFA(nn.Module):
    # Global Feature Aggregation
    def __init__(self, in_channels, squeeze_ratio=4):
        super(GFA, self).__init__()
        inter_channels = in_channels // squeeze_ratio  # reduce computation load
        self.conv_q = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.delta = nn.Parameter(torch.Tensor([0.1]))  # initiate as 0.1
        self.cr = ChannelRecalibration(in_channels)

    def forward(self, ftr):
        B, C, H, W = ftr.size()
        P = H * W
        ftr_q = self.conv_q(ftr).view(B, -1, P).permute(0, 2, 1)  # [B, P, C']
        ftr_k = self.conv_k(ftr).view(B, -1, P)  # [B, C', P]
        ftr_v = self.conv_v(ftr).view(B, -1, P)  # [B, C, P]
        weights = F.softmax(torch.bmm(ftr_q, ftr_k), dim=1)  # column-wise softmax, [B, P, P]
        G = torch.bmm(ftr_v, weights).view(B, C, H, W)
        out = self.delta * G + ftr
        out_cr = self.cr(out)
        return out_cr


class GCA(nn.Module):
    # Global Context-aware Attention
    def __init__(self, in_channels, use_pyramid):
        super(GCA, self).__init__()
        assert isinstance(use_pyramid, bool)
        self.use_pyramid = use_pyramid
        self.gfa = GFA(in_channels)
        if self.use_pyramid:
            self.cpa = CPA(in_channels)
        else:
            self.sau = SpatialAttention()

    def forward(self, ftr):
        ftr_global = self.gfa(ftr)
        if self.use_pyramid:
            att, ftr_refined = self.cpa(ftr_global)
            return att, ftr_refined
        else:
            att = self.sau(ftr_global)
            return att, ftr_global


class AttentionFusion(nn.Module):
    def __init__(self, num_att_maps):
        super(AttentionFusion, self).__init__()
        dim = 256
        self.conv_1 = ConvBlock(num_att_maps, dim, 3, False, 'ReLU')
        self.conv_2 = ConvBlock(dim, dim, 3, False, 'ReLU')
        self.conv_3 = ConvBlock(dim, 1, 3, False, 'Sigmoid')

    def forward(self, concat_att_maps):
        fusion_att_maps = self.conv_3(self.conv_2(self.conv_1(concat_att_maps)))
        return fusion_att_maps


class ConvBlock(nn.Module):
    def __init__(self, ic, oc, ks, use_bn, nl):
        # ic: input channels
        # oc: output channels
        # ks: kernel size
        # use_bn: True or False
        # nl: type of non-linearity, 'Non' or 'ReLU' or 'Sigmoid'
        super(ConvBlock, self).__init__()
        assert ks in [1, 3, 5, 7]
        assert isinstance(use_bn, bool)
        assert nl in ['Non', 'ReLU', 'Sigmoid']
        self.use_bn = use_bn
        self.nl = nl
        if ks == 1:
            self.conv = nn.Conv2d(ic, oc, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(ic, oc, kernel_size=ks, padding=(ks - 1) // 2, bias=False)
        if self.use_bn == True:
            self.bn = nn.BatchNorm2d(oc)
        if self.nl == 'ReLU':
            self.ac = nn.ReLU(inplace=True)
        if self.nl == 'Sigmoid':
            self.ac = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        if self.use_bn == True:
            y = self.bn(y)
        if self.nl != 'Non':
            y = self.ac(y)
        return y


class DecodingUnit(nn.Module):
    def __init__(self, deep_channels, shallow_channels, dec_channels, inter_ks, is_upsample):
        super(DecodingUnit, self).__init__()
        assert isinstance(is_upsample, bool)
        self.is_upsample = is_upsample
        self.unit_conv = ConvBlock(deep_channels + shallow_channels, dec_channels, 1, True, 'ReLU')
        conv_1 = ConvBlock(dec_channels, dec_channels // 4, 3, True, 'ReLU')
        conv_2 = ConvBlock(dec_channels // 4, dec_channels // 4, inter_ks, True, 'ReLU')
        conv_3 = ConvBlock(dec_channels // 4, dec_channels, 3, True, 'ReLU')
        self.bottle_neck = nn.Sequential(conv_1, conv_2, conv_3)

    def forward(self, deep_ftr, shallow_ftr):
        if self.is_upsample:
            deep_ftr = US2(deep_ftr)
        concat_ftr = torch.cat((deep_ftr, shallow_ftr), dim=1)
        inter_ftr = self.unit_conv(concat_ftr)
        dec_ftr = self.bottle_neck(inter_ftr)
        return dec_ftr


class SalHead(nn.Module):
    def __init__(self, in_channels, inter_ks):
        super(SalHead, self).__init__()
        self.conv_1 = ConvBlock(in_channels, in_channels // 2, inter_ks, False, 'ReLU')
        self.conv_2 = ConvBlock(in_channels // 2, in_channels // 2, 3, False, 'ReLU')
        self.conv_3 = ConvBlock(in_channels // 2, in_channels // 8, 3, False, 'ReLU')
        self.conv_4 = ConvBlock(in_channels // 8, 1, 1, False, 'Sigmoid')

    def forward(self, dec_ftr):
        dec_ftr_ups = US2(dec_ftr)
        outputs = self.conv_4(self.conv_3(self.conv_2(self.conv_1(dec_ftr_ups))))
        return outputs


class Encoder(nn.Module):
    def __init__(self, init_path):
        super(Encoder, self).__init__()
        ch = [64, 128, 256, 512, 512]
        dr = [2, 4, 8, 16, 16]
        bb = torch.load(os.path.join(init_path, 'bb.pth'))
        self.C1, self.C2, self.C3, self.C4, self.C5 = bb.C1, bb.C2, bb.C3, bb.C4, bb.C5
        self.GCA2 = GCA(ch[1], True)
        self.GCA3 = GCA(ch[2], True)
        self.GCA4 = GCA(ch[3], False)
        self.GCA5 = GCA(ch[4], False)
        self.AF3 = AttentionFusion(2)
        self.AF4 = AttentionFusion(3)
        self.AF5 = AttentionFusion(4)

    def forward(self, Im):  # [3, 224, 224]
        # stage-1
        F1 = self.C1(Im)  # [64, 112, 112]
        # stage-2
        F2 = self.C2(F1)  # [128, 56, 56]
        A2, F2 = self.GCA2(F2)  # [1, 56, 56] & [128, 56, 56]
        F2 = RC(F2, A2)  # [128, 56, 56]
        # stage-3
        F3 = self.C3(F2)  # [256, 28, 28]
        A3, F3 = self.GCA3(F3)  # [1, 28, 28] & [256, 28, 28]
        A3 = self.AF3(torch.cat([A3, DS2(A2)], dim=1))  # [1, 28, 28]
        F3 = RC(F3, A3)  # [256, 28, 28]
        # stage-4
        F4 = self.C4(F3)  # [512, 14, 14]
        A4, F4 = self.GCA4(F4)  # [1, 14, 14] & [512, 14, 14]
        A4 = self.AF4(torch.cat([A4, DS2(A3), DS4(A2)], dim=1))  # [1, 14, 14]
        F4 = RC(F4, A4)  # [512, 14, 14]
        # stage-5
        F5 = self.C5(F4)  # [512, 14, 14]
        A5, F5 = self.GCA5(F5)  # [1, 14, 14] & [512, 14, 14]
        A5 = self.AF5(torch.cat([A5, A4, DS2(A3), DS4(A2)], dim=1))  # [1, 14, 14]
        F5 = RC(F5, A5)  # [512, 14, 14]
        # F1: [B, 64, 112, 112]
        # F2: [B, 128, 56, 56]
        # F3: [B, 256, 28, 28]
        # F4: [B, 512, 14, 14]
        # F5: [B, 512, 14, 14]
        return F1, F2, F3, F4, F5


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        ch = [64, 128, 256, 512, 512]
        dr = [2, 4, 8, 16, 16]
        dec_ch = [256, 384, 512, 768]
        self.du_5 = DecodingUnit(ch[4], ch[3], dec_ch[3], 3, False)
        self.du_4 = DecodingUnit(dec_ch[3], ch[2], dec_ch[2], 5, True)
        self.du_3 = DecodingUnit(dec_ch[2], ch[1], dec_ch[1], 5, True)
        self.du_2 = DecodingUnit(dec_ch[1], ch[0], dec_ch[0], 7, True)

    def forward(self, F1, F2, F3, F4, F5):
        D4 = self.du_5(F5, F4)
        D3 = self.du_4(D4, F3)
        D2 = self.du_3(D3, F2)
        D1 = self.du_2(D2, F1)
        # D1: [256, 112, 112]
        # D2: [384, 56, 56]
        # D3: [512, 28, 28]
        # D4: [768, 14, 14]
        return D1, D2, D3, D4


class DAFNet(nn.Module):
    def __init__(self, return_loss, init_path):
        super(DAFNet, self).__init__()
        ch = [64, 128, 256, 512, 512]
        dr = [2, 4, 8, 16, 16]
        dec_ch = [256, 384, 512, 768]
        self.return_loss = return_loss
        self.encoder = Encoder(init_path)
        initiate(self.encoder, os.path.join(init_path, 'ei.pth'))
        self.decoder = Decoder()
        initiate(self.decoder, os.path.join(init_path, 'di.pth'))
        mh = SalHead(dec_ch[0], 7)
        eh = SalHead(dec_ch[0], 7)
        self.head = nn.ModuleList([mh, eh])
        self.bce = nn.BCELoss()

    def forward(self, image, label, edge):
        F1, F2, F3, F4, F5 = self.encoder(image)
        D1, D2, D3, D4 = self.decoder(F1, F2, F3, F4, F5)
        sm = self.head[0](D1)
        se = self.head[1](D1)
        if self.return_loss:
            losses_list = self.compute_loss(sm, se, label, edge)
            return sm, se, losses_list
        else:
            return sm, se

    def compute_loss(self, sm, se, label, edge):
        mask_loss = self.bce(sm, label)
        edge_loss = self.bce(se, edge)
        total_loss = 0.7 * mask_loss + 0.3 * edge_loss
        return [total_loss, mask_loss, edge_loss]



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

    img_idx = '0113'  # 0257 0092 0388 0354 0384
    image_path = f'./data/ORSSD/train/image/{img_idx}.jpg'
    gt_path = f'./data/ORSSD/train/gt/{img_idx}.png'
    edge_path = f'../data/ORSSD/train/emfi_edge/{img_idx}.png'

    image = Image.open(image_path).convert('RGB')
    img = preprocess(image).to(device).unsqueeze(0)
    gt = pil2tensor(Image.open(gt_path).convert('L')).to(device).unsqueeze(0)
    edge = pil2tensor(Image.open(gt_path).convert('L')).to(device).unsqueeze(0)

    loss = 0.0
    model = DAFNet(loss).to(device)
    # weight = 'models/CEHAFNet_ORSSD/20250428_095305-73659e1c-5CEk5p2NoReLU-GuideNoBR+UpConv-NmadpPRI-0.1e-4-6/LowestLoss.pth'
    # model.load_state_dict(torch.load(weight, map_location='cuda'))
    # model.eval()

    out = model(img, gt, edge)
    output = out[0]
    img2show = output.squeeze(0).cpu()
    output_img = tensor2pil(img2show)
    output_img.show()

    flops, param = profile(model, [img, ])
    flops, param = clever_format([flops, param], "%.2f")
    print("flops are :" + flops, "params are :" + param)

    tb_dir = 'nets/tensorboard/temp'
    Path(tb_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=f'{tb_dir}')
    writer.add_graph(model, img)
    writer.close()

import warnings
from collections import deque
from typing import Tuple
import torch
from PIL import Image
from softadapt.base._softadapt_base_class import SoftAdaptBase
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from softadapt import LossWeightedSoftAdapt, SoftAdapt, NormalizedSoftAdapt
from torch.autograd import Variable
import numpy as np
from math import exp
import wandb
from torch.nn import Module, MSELoss
from torchvision import transforms
from torchvision.transforms import v2

# from libraries.pytorch_iou import IOU as IoU0
# from pytorch_msssim import ssim as ssim1
from scipy.optimize import minimize
import kornia.contrib as kc

# ---------- 改进数值不稳定 ----------
class IOULoss(nn.Module):
    def __init__(self, soft=False, alpha=False, size_average=True, epsilon=1e-8):
        super(IOULoss, self).__init__()
        self.soft = soft
        self.alpha = alpha
        self.size_average = size_average
        self.epsilon = epsilon

    def forward(self, pred, target):
        return _iou_loss(pred, target, self.soft, self.alpha, self.size_average, self.epsilon)


def _iou_loss(pred, target, soft=False, alpha=False, size_average=True, epsilon=1e-8):
    if soft:
        pred = torch.sigmoid(pred)
    if alpha:
        pred_sum = pred.sum(dim=(1, 2, 3))
        target_sum = target.sum(dim=(1, 2, 3))
        dis = torch.pow((pred_sum - target_sum) / 2, 2)
        alpha = (torch.min(pred_sum, target_sum) + dis + epsilon) / (torch.max(pred_sum, target_sum) + dis + epsilon)
    else:
        alpha = 1

    intersection = (pred * target).sum(dim=(1, 2, 3))  # Intersection over the batch
    union = (pred + target).sum(dim=(1, 2, 3)) - intersection  # Union over the batch

    iou = (intersection + epsilon) / (union + epsilon)  # Compute IoU

    iou_loss = 1 - alpha * iou  # IoU loss is 1 - IoU

    if size_average:
        return iou_loss.mean()  # Return average loss over the batch
    else:
        return iou_loss.sum()  # Return sum loss over the batch


# MENet Soft IOU
def soft_iou(preds, target):
    pred = torch.sigmoid(preds)

    inter = torch.sum(target * pred, dim=(1, 2, 3))
    union = torch.sum(target, dim=(1, 2, 3)) + torch.sum(pred, dim=(1, 2, 3)) - inter
    soft_iou_loss = 1 - ((inter + 1) / (union + 1)).mean()
    return soft_iou_loss


class RegionLoss(nn.Module):
    def __init__(self, k1=0.01, k2=0.03, l=1, epsilon=1e-10, sig=False, log=False, size_average=True):
        super(RegionLoss, self).__init__()
        self.k1, self.k2, self.k3 = k1, k2, k2/2
        self.l, self.epsilon = l, epsilon
        self.sig = sig
        self.log, self.size_average = log, size_average
        self.alpha = 1
        self.beta = 1
        self.gamma = 1

    def centroid(self, gt):
        b, c, h, w = gt.shape
        area_object = gt.sum(dim=(2, 3))  # Shape (B, C)

        safe_area = area_object + self.epsilon * (area_object == 0).float()     # 添加数值稳定性保护

        row_ids = torch.arange(h, device=gt.device).view(1, 1, h, 1).expand(b, c, h, w)
        col_ids = torch.arange(w, device=gt.device).view(1, 1, 1, w).expand(b, c, h, w)

        x = (gt * col_ids).sum(dim=(2, 3)) / safe_area
        y = (gt * row_ids).sum(dim=(2, 3)) / safe_area

        return x.round().long().clamp(0, w - 1), y.round().long().clamp(0, h - 1)

    def img2region(self, pred, gt, centroid):
        x, y = centroid  # x, y shape (B, C)
        b, c, h, w = gt.shape
        area_total = h * w

        x = x.view(b, c, 1, 1)
        y = y.view(b, c, 1, 1)

        rows = torch.arange(h, device=gt.device).view(1, 1, h, 1).expand(b, c, h, w)
        cols = torch.arange(w, device=gt.device).view(1, 1, 1, w).expand(b, c, h, w)

        mask_LT = (rows < y) & (cols < x)
        mask_RT = (rows < y) & (cols >= x)
        mask_LB = (rows >= y) & (cols < x)
        mask_RB = (rows >= y) & (cols >= x)

        masks = [mask.float() for mask in [mask_LT, mask_RT, mask_LB, mask_RB]]

        w1 = torch.clamp((x * y) / area_total, min=0, max=1)
        w2 = torch.clamp((y * (w - x)) / area_total, min=0, max=1)
        w3 = torch.clamp(((h - y) * x) / area_total, min=0, max=1)
        w4 = torch.clamp(1 - w1 - w2 - w3, min=0, max=1)
        weights = [w1, w2, w3, w4]

        return {"masks": masks, "weights": weights}

    def ssim(self, pred, gt, mask):
        b, c, h, w = gt.shape
        area = mask.sum(dim=(2, 3)) + self.epsilon

        # 均值计算改进
        valid_mask = (area > self.epsilon)
        mean_pred = torch.where(valid_mask,
                                (pred * mask).sum(dim=(2, 3)) / area,
                                torch.zeros_like(area))
        mean_gt = torch.where(valid_mask,
                              (gt * mask).sum(dim=(2, 3)) / area,
                              torch.zeros_like(area))

        delta_pred = pred - mean_pred.view(b, c, 1, 1)
        delta_gt = gt - mean_gt.view(b, c, 1, 1)

        # 方差协方差计算改进
        var_pred = (delta_pred ** 2 * mask).sum(dim=(2, 3)) / (area + self.epsilon)
        var_gt = (delta_gt ** 2 * mask).sum(dim=(2, 3)) / (area + self.epsilon)
        cov_xy = (delta_pred * delta_gt * mask).sum(dim=(2, 3)) / (area + self.epsilon)

        # 稳定性增强
        std_pred = torch.sqrt(var_pred + self.epsilon)
        std_gt = torch.sqrt(var_gt + self.epsilon)

        c1 = (self.k1 * self.l) ** 2
        c2 = (self.k2 * self.l) ** 2
        c3 = (self.k3 * self.l) ** 2

        # 分量计算改进
        luma = (2 * mean_pred * mean_gt + c1) / (mean_pred ** 2 + mean_gt ** 2 + c1 + self.epsilon)
        contrast = (2 * std_pred * std_gt + c2) / (var_pred + var_gt + c2 + self.epsilon)
        structure = torch.clamp((cov_xy + c3) / (std_pred * std_gt + c3 + self.epsilon), min=-1.0, max=1.0)

        ssim_val = (luma ** self.alpha) * (contrast ** self.beta) * (structure ** self.gamma)
        return torch.clamp(ssim_val, 0.0, 1.0)

    def forward(self, pred, gt):
        pred = torch.sigmoid(pred) if self.sig else pred
        centroid = self.centroid(gt)
        region = self.img2region(pred, gt, centroid)
        masks, weights = region["masks"], region["weights"]

        total_loss = 0
        for mask, weight in zip(masks, weights):
            # 区域有效性检查
            valid = (mask.sum(dim=(1, 2, 3)) > self.epsilon).float()
            ssim_region = self.ssim(pred, gt, mask)

            if self.log:
                ssim_region = torch.clamp(ssim_region, min=self.epsilon, max=1 - self.epsilon)
                loss_term = -torch.log(ssim_region) * valid
            else:
                loss_term = (1 - ssim_region) * valid

            total_loss += (loss_term * weight).sum(dim=1)

        if self.size_average:
            return total_loss.mean()
        else:
            return total_loss.sum()




class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: 网络输出的概率图，shape为[B,1,H,W]或[B,H,W]，已经过sigmoid
            target: 真实标签，shape同pred，值为0或1
        Returns:
            dice_loss: 计算得到的损失值
        """
        # 确保输入维度一致
        pred = pred.view(-1)
        target = target.view(-1)

        # 计算交集
        intersection = (pred * target).sum()

        # 计算dice系数
        dice = (2. * intersection + self.smooth) / (
                pred.sum() + target.sum() + self.smooth
        )

        # 返回dice损失
        return 1 - dice


class DiceLossv2(nn.Module):

    def __init__(self, sig=False, eps=1e-8):  # 添加 c 参数
        super(DiceLossv2, self).__init__()
        self.eps = eps
        self.sig = sig

    def forward(self, pred, target):
        if self.sig:
            pred = torch.sigmoid(pred)
        sum_pg = torch.sum(pred * target, dim=(2, 3))
        sum_p2 = torch.sum(pred ** 2, dim=(2, 3))
        sum_g2 = torch.sum(target ** 2, dim=(2, 3))

        numerator = 2 * sum_pg + self.eps  # 组合项
        denominator = sum_p2 + sum_g2 + self.eps

        em_global = numerator / denominator
        loss = 1 - em_global.mean()
        return loss


class StableKLLoss(nn.Module):
    def __init__(self, eps=1e-10, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.kldiv = nn.KLDivLoss(reduction=reduction)

    def forward(self, pred, gt):
        """
        Args:
            pred (Tensor): 预测值 (B,1,H,W) in [0,1] (经过 Sigmoid)
            gt (Tensor): 目标值 (B,1,H,W) in [0,1]
        Returns:
            loss (Tensor): 标量
        """
        # 输入检查
        assert torch.all(pred >= 0.0) and torch.all(pred <= 1.0), "pred 值域需在 [0,1] 内"
        assert torch.all(gt >= 0.0) and torch.all(gt <= 1.0), "gt 值域需在 [0,1] 内"

        # 将 pred 转换为对数概率分布
        pred = torch.clamp(pred, min=self.eps, max=1.0)  # 避免 log(0)
        log_pred = torch.log(pred)  # 对数概率分布

        # 将 gt 归一化为概率分布
        gt = gt / (gt.sum(dim=(1, 2, 3), keepdim=True) + self.eps)  # 归一化并避免除零

        # 计算 KL 散度损失
        loss = self.kldiv(log_pred, gt)
        return loss


class StructureLoss(nn.Module):
    """ SARNet """
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


class NormalizedLossWeightedSoftAdapt(SoftAdaptBase):
    """Class implementation of the normalized-loss-weighted SoftAdapt variant.

    Attributes:
        beta: A float hyperparameter controlling weight distribution.
            - beta > 0: Assign higher weights to poorly performing components.
            - beta < 0: Assign higher weights to well-performing components.
            - beta = 0: Equal weights (trivial case).
        accuracy_order: Order of finite difference approximation for slopes.
    """
    _FIRST_ORDER_COEFFICIENTS = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    _THIRD_ORDER_COEFFICIENTS = torch.tensor([-11 / 6, 3.0, -3 / 2, 1 / 3], dtype=torch.float32)
    _FIFTH_ORDER_COEFFICIENTS = torch.tensor([-137 / 60, 5.0, -5.0, 10 / 3, -5 / 4, 1 / 5], dtype=torch.float32)

    def __init__(self, beta: float = 0.1, accuracy_order: int = 5, eps=1e-10):
        """Initialize the variant."""
        super().__init__()
        self.beta = beta
        self.accuracy_order = accuracy_order
        self.eps = eps

    def _get_finite_difference(
            self,  # 添加 self 参数
            input_tensor: torch.Tensor,
            order: int = None,
            verbose: bool = True
    ) -> torch.Tensor:
        """PyTorch 版本的有限差分计算函数."""
        # 检查输入长度与阶数的兼容性
        if order is None:
            order = len(input_tensor) - 1
            if verbose:
                print(f"==> 未指定有限差分的阶数，自动推断为 {order}（输入长度 {len(input_tensor)}）。")

        if order > len(input_tensor):
            raise ValueError("有限差分阶数不能超过输入张量的长度。")

        # 截取需要的输入数据段
        if order + 1 < len(input_tensor):
            if verbose:
                print(f"==> 输入长度 {len(input_tensor)} 超过阶数+1 ({order + 1})，使用最后 {order + 1} 个元素。")
            input_tensor = input_tensor[-(order + 1):]

        # 加载预定义的差分系数
        device = input_tensor.device  # 保持设备一致性
        if order % 2 == 0 and order > 5:
            raise ValueError("阶数大于5时必须为偶数。")

        # 加载预定义的差分系数（通过类属性）
        if order == 1:
            constants = self._FIRST_ORDER_COEFFICIENTS.to(device)
        elif order == 3:
            constants = self._THIRD_ORDER_COEFFICIENTS.to(device)
        elif order == 5:
            constants = self._FIFTH_ORDER_COEFFICIENTS.to(device)
        else:
            raise NotImplementedError(f"暂不支持阶数 {order} 的差分计算。")

        # 确保输入张量与系数长度匹配
        if len(input_tensor) != len(constants):
            raise ValueError(f"输入长度 ({len(input_tensor)}) 与系数长度 ({len(constants)}) 不匹配。")

        ### TODO
        

    def _compute_rates_of_change(
            self,
            input_tensor: torch.Tensor,
            order: int = 5,
            verbose: bool = True
    ) -> torch.Tensor:
        ### TODO 

    def get_component_weights(self,
                              *loss_component_values: Tuple[torch.Tensor],
                              verbose: bool = True):
        """Compute weights combining normalized slopes and loss magnitudes.

        Args:
            loss_component_values: Historical values of each loss component.
            verbose: Flag to enable warnings.

        Returns:
            torch.Tensor: Normalized-loss-weighted component weights.
        """
        if len(loss_component_values) == 1:
            print("Warning: Trivial weighting with single loss component.")

        device = loss_component_values[0].device  # 获取输入张量的设备
        rates_of_change = []
        average_loss_values = []

        # Step 1: 计算变化率和平均损失值
        for loss_points in loss_component_values:
            loss_points = loss_points.detach()  # 仅分离梯度，保持设备不变
            computed_rates = self._compute_rates_of_change(loss_points, self.accuracy_order, verbose)
            rates_of_change.append(computed_rates)
            average_loss_values.append(torch.mean(loss_points))

        # 合并张量（自动保持设备一致性）
        slopes = torch.stack(rates_of_change)
        average_loss_values = torch.stack(average_loss_values)

        # Step 2: 归一化变化率
        ns_i = slopes / (torch.sum(torch.abs(slopes)) + self.eps)
        rates_of_change_normalized = ns_i - torch.max(ns_i)

        # Step 3: 计算加权 Softmax
        weights = self._softmax(
            input_tensor=rates_of_change_normalized,
            beta=self.beta,
            numerator_weights=average_loss_values
        )

        return weights


class AdpMLL(nn.Module):
    def __init__(self, sftAdpType, device, loss_components:list, supervised_nums=6, update_frequency=6, accuracy_order=5, beta=0.1):
        super(AdpMLL, self).__init__()
        """
        :param sftAdpt:
            Attributes:
                'SoftAdapt': 只关注每个损失函数的变化率
                'NormalizedSoftAdapt': 对斜率进行归一化，显著减少变化率的差异，从而使三个分量之间的权重分布更加均匀
                'LossWeightedSoftAdapt': 考虑变化率，还考虑损失函数的值
                'NormalizedWeightedSoftAdapt': 
        """
        # === 参数校验 ===
        assert update_frequency >= accuracy_order + 1, "update_frequency必须≥accuracy_order+1"
        assert sftAdpType in [SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt, NormalizedLossWeightedSoftAdapt],  \
            "there isn't this sftAdpt！[SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt, NormalizedLossWeightedSoftAdapt]"

        # === 核心参数 ===
        self.loss_components = loss_components
        self.supervised_nums = supervised_nums
        self.update_frequency = update_frequency
        self.accuracy_order = accuracy_order
        self.beta = beta
        self.device = device
        self.training_mode = True
        self.sftAdpType = sftAdpType

        # === 自适应权重初始化 ===
        n = len(self.loss_components)   # 损失组件的数量
        self.softadapt_lst = [
            self.sftAdpType(beta=self.beta, accuracy_order=self.accuracy_order)
            for _ in range(self.supervised_nums)
        ]

        ## 初始化权重1
        # self.adapt_weights = [
        #     torch.ones(len(self.loss_components), device=self.device)
        #     for _ in range(self.supervised_nums)
        # ]
        ## 初始化权重1/n
        self.adapt_weights = [
            torch.full((n,), 1.0 / n, device=self.device)
            for _ in range(self.supervised_nums)
        ]
        self.current_tmp_weights = [w.clone() for w in self.adapt_weights]

        # === 历史损失记录（长度限制为accuracy_order+1） ===
        self.loss_values_history = [
            [deque(maxlen=self.accuracy_order + 1) for _ in range(len(self.loss_components))]
            for _ in range(self.supervised_nums)
        ]
        self.num_iter = 1

    def train(self, mode: bool = True):
        """设置模块为训练或评估模式"""
        self.training_mode = mode

    def eval(self):
        """设置模块为评估模式"""
        self.train(False)

    def get_current_weights(self):
        """返回当前各层的权重字典"""
        weight_dict = {}
        for layer_idx, weights in enumerate(self.adapt_weights):
            for loss_idx, w in enumerate(weights):
                # weight_dict[f"layer{layer_idx}/loss{loss_idx}_w"] = w.item()
                weight_dict[f"layer{layer_idx}_𝛼{loss_idx+1}"] = w.item()
        return weight_dict

    def forward(self, pred, gts):
        ## Release after article acceptance

        return total_loss

    def _update_history(self, layer_idx: int, loss_fn: nn.Module, loss_value: torch.Tensor):
        loss_idx = self.loss_components.index(loss_fn)
        self.loss_values_history[layer_idx][loss_idx].append(loss_value.detach().cpu())

    def _update_weights(self, layer_idx):
        # 检查所有损失分量的历史记录是否已填满
        # 检查历史数据是否足够
        all_ready = all(
            len(hist) >= self.accuracy_order + 1
            for hist in self.loss_values_history[layer_idx]
        )
        if not all_ready:
            return

        # 截取最后 (accuracy_order+1) 个点
        loss_histories = [
            torch.tensor(list(hist)[-self.accuracy_order - 1:])
            for hist in self.loss_values_history[layer_idx]
        ]
        # 计算新权重
        new_weights = self.softadapt_lst[layer_idx].get_component_weights(*loss_histories)
        self.adapt_weights[layer_idx] = new_weights.to(self.device)


bce_loss = nn.BCELoss(size_average=True)    # 输入需要经过sigmoid
# bce_loss = nn.BCEWithLogitsLoss(size_average=True)
f_loss = FLoss(beta=0.3, log_like=False)
iou_loss = IOULoss(soft=False, size_average=True)
ssim_value = SSIM(window_size=11, size_average=True)
logssim_value = LOGSSIM(window_size=11, size_average=True)
# pixel_loss = nn.MSELoss(reduction='mean')
region_loss = RegionLoss(l=1)
# region_loss = ContrastLoss()
# image_loss = nn.MSELoss(reduction='mean')
# image_loss = DiceLoss()
image_loss = IOULoss(soft=False, size_average=True)
# image_loss = FLoss()



if __name__ == '__main__':
    pil2tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    tensor2pil = v2.ToPILImage()
    preprocess = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((256, 256)),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_idx = '0257'
    image_path = f'/home/lsa/Shared/data/ORSSD/test/image/{img_idx}.jpg'
    gt_path = f'/home/lsa/Shared/data/ORSSD/test/gt/{img_idx}.png'

    image = Image.open(image_path).convert('RGB')
    gt = pil2tensor(Image.open(gt_path)).unsqueeze(0)
    gt = transforms.Resize((256,256))(gt)
    img = preprocess(image).unsqueeze(0)  # 因为在batchnorm会检查类型，必须是4D

    conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
    img = torch.sigmoid(conv(img))

    region = RegionLoss()
    # l1_loss = L1Loss()
    # l1 = nn.L1Loss(reduction='mean')
    # l2 = nn.MSELoss(reduction='mean')
    # histogramloss = HistogramLoss()
    # cos = CosineSimilarityLoss()
    # kl_loss = nn.KLDivLoss(reduction='mean')  # reduction: 'none'|'batchmean'|'sum'
    # kl = StableKLLoss()
    # adppri = AdaptBIS()
    g_loss = DiceLossv2()
    regionv2 = RegionLoss_v2()

    # print('RegionSSIMLoss', region(img, gt))
    # print('Histogram Loss', histogramloss(img, gt))
    # print('L2_loss', l2(img, gt))
    # print('iou_loss', _iou_loss(img, gt))
    # print('w_iou_loss', _iou_loss(img, gt, alpha=True))
    # print('cos_sim_loss', cos(img, gt))
    # print('kl_loss', kl(img, gt))
    # print('kl_nn', kl_loss(img, gt))
    # print('adppri:', adppri(img, gt))
    print('g_loss', g_loss(img, gt))
    print('regionv1', region(img, gt))
    print('regionv2', regionv2(img, gt))
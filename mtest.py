import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import *
from PIL import Image
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from nets.CEHAFNet.ablation_r import CEHAFNet
from utils.data import SalObjDataset
from utils.transforms import RescaleT, ToTensorLab
import torch.nn.functional as F


def MaxMinNormalization(x):
    Max = torch.max(x)
    Min = torch.min(x)
    x = torch.div(torch.sub(x, Min), 0.0001 + torch.sub(Max, Min))
    return x


if __name__ == '__main__':
    # --------- 1. get image path and name ---------
    net = CEHAFNet()
    dataset = 'EORSSD'
    # challenge_type = 'challenge/OC'
    image_dir = f'./data/{dataset}/test/image/'
    # prediction_dir = f'/home/lsa/Shared/MatlabEvaluation/SalMap/{net.__class__.__name__}/{dataset}/'
    prediction_dir = f'./SalRedo/{net.__class__.__name__}/{dataset}_r/'
    # image_dir = f'./data/{dataset}/{challenge_type}/images/'
    # prediction_dir = f'/home/lsa/Shared/MatlabEvaluation/SalMap/{net.__class__.__name__}/{challenge_type}/'
    os.makedirs(prediction_dir, exist_ok=True)

    # 0：GPU 1：CPU
    device = '0'
    device = torch.device("cuda" if torch.cuda.is_available() and device == '0' else "cpu")
    ckpt2weights = False

    # 0: False  1:Sigmoid  2:MaxMinNorm
    pred2norm = 0
    isdiff = True
    COLOR_MAP = torch.tensor([
        [0, 0, 0],  # TN (黑)
        [255, 99, 71],  # FP (红)
        [60, 179, 113],  # FN (绿)
        [255, 255, 255]  # TP (白)
    ], dtype=torch.uint8, device=device)

    # 从断点恢复模型权重 xx_state_dict
    if ckpt2weights:
        ckpt_path = str(input('input ckpt_path: '))
        ckpt = torch.load(ckpt_path)
        epoch = ckpt['epoch']
        model_weight = ckpt['model_state_dict']
        model_dir = f'./models/{net.__class__.__name__}/ckpt2model'
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model_path = model_dir + f'/{dataset}-ckpt2model-{epoch}.pth'
        torch.save(model_weight, model_path)
    else:
        # 从模型权重恢复模型
        model_path = str(input('model_path: '))

    # --------- 2. dataloader ---------
    #1. dataload
    img_name_list = glob.glob(image_dir + '*.jpg')
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, gt_name_list=[],
                                        transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]))
    test_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=3)

    # --------- 3. model define ---------
    print("\n...load weight...")
    net.to(device)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    # --------- 4. inference for each image ---------
    time_sum = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    pbar = tqdm(test_dataloader, total=len(test_salobj_dataset), desc='Inference')
    for i_test, data_test in enumerate(pbar):
        img_path = img_name_list[i_test]
        original_w, original_h = Image.open(img_path).size

        inputs, gts = data_test['image'], data_test['label']
        inputs = inputs.to(device)
        gts = gts.to(device)

        start.record()
        outputs = net(inputs)
        end.record()
        torch.cuda.synchronize()
        time_once = start.elapsed_time(end)
        time_sum += time_once

        d1 = outputs[0]
        pred = F.interpolate(d1, size=(original_h, original_w), mode='bilinear', align_corners=False)

        if pred2norm == 1:
            pred = torch.sigmoid(pred)
        elif pred2norm == 2:
            pred = MaxMinNormalization(pred)
        else:
            pass

        # if isdiff:
        #     gts = F.interpolate(gts.float(), size=(original_h, original_w), mode='bilinear', align_corners=False)
        #     diff_map = 2 * gts + pred
        #     COLOR_MAP_BC = COLOR_MAP.view(1, 4, 1, 1, 3)
        #
        #     # 直接映射颜色 (B,1,H,W,3)
        #     diff_rgb = COLOR_MAP_BC[:, diff_map.long(), :, :, :]
        #
        #     # 调整维度 (B,3,H,W)
        #     diff_rgb = diff_rgb.squeeze(1).permute(0, 3, 1, 2)
        #     pred = diff_rgb

        # 构造输出文件名（保留原名但改成 .png）
        base_name = os.path.splitext(os.path.basename(img_name_list[i_test]))[0]
        out_path = os.path.join(prediction_dir, base_name + '.png')
        save_image(pred, out_path, format='PNG')
        pbar.set_postfix({
            f'No.{i_test + 1}': f'{base_name}'
        })

    avg_time = time_sum / len(test_dataloader)
    print(f'Running time:{time_sum}, FPS:{1000/avg_time}')
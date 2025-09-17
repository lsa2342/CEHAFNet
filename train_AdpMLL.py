import glob
import logging
import sys
from datetime import datetime
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MetricCollection, FBetaScore, MeanAbsoluteError
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM_metric
from torchvision.transforms import v2 as transforms
from tqdm import tqdm
import wandb
import warnings

warnings.filterwarnings(action="ignore")

from utils.data import SalObjDataset
from utils.emeasure import EMeasure
from utils.loss import bce_loss, region_loss, image_loss, NormalizedLossWeightedSoftAdapt, AdaptPRI
from utils.transforms import RandomCrop, RescaleT, ToTensorLab, ColorJitter, RandomShadow, Rescale
from utils.utils import load_checkpoint, get_logger, random_seed
from nets.CEHAFNet.ablation_r import CEHAFNet

debug_flag = sys.gettrace()
torch.autograd.set_detect_anomaly(True)  # 启用自动微分异常检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not debug_flag else "cpu"
model = CEHAFNet(3, 1).to(device)
wandb_mode = 'offline' #'online' if not debug_flag else 'offline'  #'offline'
metric_collection = MetricCollection({
    'SSIM': SSIM_metric(),
    'Fβ': FBetaScore(task="binary", beta=0.3),
    'Eξ': EMeasure(),  # MyORSISOD自定义
    'MAE': MeanAbsoluteError(),
    # 'PR': PrecisionRecallCurve(task="binary", thresholds=[float(i) / 255.0 for i in range(2)])
}).to(device)  # metrics calculator

# ---------------------------------------------------------------------------- #
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
epoch_num = 100
batch_size_train = 16
batch_size_val = 16
num_workers = 6
ckpt_interval = 3
is_resume = False
project = f'{model.__class__.__name__}' if not debug_flag else f'test'
target = ' '
describe = f'{target}_'
random_seed(2342)

dataset = 'EORSSD'
dataset_dir = f'./data/{dataset}'
model_dir = f"./models/{model.__class__.__name__}_{dataset}/{current_time}-{target}"
checkpoint_dir = f"./checkpoints/{model.__class__.__name__}_{dataset}/{current_time}-{target}"
tensorboard_dir = f'./tensorboard/{current_time}-{target}'

train_image_lst = sorted(glob.glob(dataset_dir + '/train/image/*.jpg'))
train_label_lst = sorted(glob.glob(dataset_dir + '/train/gt/*.png'))
val_image_lst = sorted(glob.glob(dataset_dir + '/test/image/*.jpg'))
val_label_lst = sorted(glob.glob(dataset_dir + '/test/gt/*.png'))

train_transforms = transforms.Compose([ColorJitter(0.5, 0.5, 0.5, 0.25), RandomShadow(),
                                       Rescale(256), RandomCrop(224), ToTensorLab(flag=0)])
val_transforms = transforms.Compose([RescaleT(256), ToTensorLab(flag=0)])
train_dataset = SalObjDataset(img_name_list=train_image_lst, gt_name_list=train_label_lst, transform=train_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
val_dataset = SalObjDataset(img_name_list=val_image_lst, gt_name_list=val_label_lst, transform=val_transforms)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-7)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
criterion = AdaptPRI(NormalizedLossWeightedSoftAdapt,
                     device='cuda', loss_components=[bce_loss, region_loss, image_loss],
                     update_frequency=6, beta=-0.1)

# ------- training process -------
best_loss = float('inf')
MAE_best = float('inf')
F_beta_best = 0.0
if not is_resume:
    start_epoch = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    run_id = wandb.util.generate_id()
else:
    model_dir = Path(str(input('请输入恢复训练保存model的目录: ')))
    checkpoint_path = Path(str(input('请输入训练记录的checkpoint路径: ')))
    tensorboard_dir = Path(str(input('请输入恢复训练的tensorboard日志目录: ')))
    run_id = str(input('恢复wandb训练记录的run_id: '))
    start_epoch, running_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    print(f"Resuming training from epoch {start_epoch}...")
Path(model_dir).mkdir(parents=True, exist_ok=True)
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(tensorboard_dir)

wandb.login(anonymous="allow", key=' ', relogin=True)
wandb.init(
    project=project,
    name=target,
    notes=describe,
    tags=[str(dataset)],
    mode=wandb_mode,
    resume="allow",
    id=run_id,
    config={
        "date": current_time,
        "backbone": "resnet50",
        "epochs": epoch_num,
        "batchsize": batch_size_train,
        "loss": "adapt",
        "learning_rate": optimizer.param_groups[0]['lr'],
    },
    # settings=wandb.Settings(start_method="fork"),
    settings=wandb.Settings(start_method="spawn"),
)
wandb.save('nets/CEHAFNet/ablation.py')  # TODO
wandb.define_metric("train/*", step_metric="train/step_train")
wandb.define_metric("val/*", step_metric="val/step_val")
wandb.watch(model)

for epoch in range(start_epoch, epoch_num):
    model.train()
    criterion.train()
    step_train = 0
    loss2tb = {}
    running_loss = running_loss if epoch == start_epoch else 0.0
    pbar = tqdm(train_dataloader, total=len(train_dataloader), desc=f'train Epoch {epoch + 1}/{epoch_num}')
    for i, data in enumerate(pbar):
        inputs, gts = data['image'], data['label']
        inputs, gts = inputs.to(device), gts.to(device)

        optimizer.zero_grad()

        preds = model(inputs)
        loss_batch = criterion(preds, gts)
        loss_batch.backward()
        optimizer.step()

        running_loss += loss_batch.item()

        step_train += 1
        train_log_dict = {
            "train/step_train": step_train + epoch * len(train_dataloader),
            "train/loss": running_loss / step_train,
            "weight/": criterion.get_current_weights(),
            "epoch": epoch + 1,
        }
        pbar.set_postfix({'loss': running_loss / step_train})
        wandb.log(train_log_dict)

    loss2tb['train'] = running_loss / len(train_dataloader)


    # Save checkpoint
    checkpoint_state = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'running_loss': running_loss,
        # 'running_tar_loss': running_tar_loss
    }
    model_logger = get_logger(filename=f'{model_dir}/model_{describe}-{current_time}.log', level=logging.INFO)
    ckpt_logger = get_logger(filename=f'{checkpoint_dir}/ckpt_{describe}-{current_time}.log', level=logging.INFO)
    if validation_loss <= best_loss:
        best_loss = validation_loss
        torch.save(model.state_dict(), f"{model_dir}/LowestLoss.pth")
        model_logger.info(
            f'Epoch: {epoch + 1}   validateLoss: {validation_loss:.4f}   BestLoss: {best_loss:.4f} Store:{model_dir}/LowestLoss.pth')
    if F_beta_val >= F_beta_best:
        torch.save(model.state_dict(), f'{model_dir}/F_beta_best.pth')
        model_logger.info(
            f'Epoch: {epoch + 1}   F_beta_val: {F_beta_val:.4f}   F_beta_best: {F_beta_best:.4f} Store:{model_dir}/F_beta_best.pth')
        F_beta_best = F_beta_val
    if MAE_val < MAE_best:
        torch.save(model.state_dict(), f'{model_dir}/mae_best.pth')
        model_logger.info(
            f'Epoch: {epoch + 1}   mae_val: {MAE_val:.4f}   MAE_best: {MAE_best:.4f} Store:{model_dir}/MAE_best.pth')
        MAE_best = MAE_val
    if epoch == 5:  # 刚开始、中间保存
        ckpt = f'{checkpoint_dir}/ckpt.pth.{epoch + 1}'
        torch.save(checkpoint_state, ckpt)
        ckpt_logger.info(
            f'Epoch: {epoch + 1}   running_loss: {running_loss:.4f}   Store:{ckpt}')
    elif epoch % ckpt_interval == 0:
        ckpt = f'{checkpoint_dir}/newest_ckpt.pth'
        torch.save(checkpoint_state, ckpt)
        ckpt_logger.info(
            f'Epoch: {epoch + 1}   running_loss: {running_loss:.4f}   Store:{ckpt}')

# --------------------------- test ----------------------------
# wandb.summary['test_Fβ'] = F_beta_val
print('-------------Congratulations! Training Done!!!-------------')

writer.close()
wandb.finish()

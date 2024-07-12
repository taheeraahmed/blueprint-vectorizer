import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from imageio import imsave, imread
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.unet import UNet
from models.github_model import *
import metric_loss
from data_loader import FloorplanDataset
from models.unet.unet_model import UNet
from utils.misc import save_checkpoint, count_parameters, transfer_optimizer_to_gpu, dice_score
from utils.config import Struct, load_config, compose_config_str, parse_arguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
args = parse_arguments()
config_dict = load_config(file_path="utils/config.yaml")
configs = Struct(**config_dict)
config_str = compose_config_str(configs, keywords=["lr", "batch_size"])
exp_dir = os.path.join(configs.exp_base_dir, "%d/" % args.test_fold_id)
configs.exp_dir = exp_dir

writer = SummaryWriter(log_dir=os.path.join(configs.exp_dir, "logs"))

ckpt_save_path = os.path.join(configs.exp_dir)
if not os.path.exists(ckpt_save_path):
    os.mkdir(ckpt_save_path)

if configs.seed:
    torch.manual_seed(configs.seed)
    if configs.use_cuda:
        torch.cuda.manual_seed_all(configs.seed)
    np.random.seed(configs.seed)
    random.seed(configs.seed)
    print("Set random seed to {}".format(configs.seed))


# Dataloader
train_dataset = FloorplanDataset("train", args.test_fold_id, configs=configs)
val_dataset = FloorplanDataset("val", args.test_fold_id, configs=configs)

train_loader = DataLoader(
    train_dataset,
    batch_size=configs.batch_size,
    num_workers=configs.num_workers,
    shuffle=True,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=configs.batch_size,
    num_workers=configs.num_workers,
    shuffle=False,
    pin_memory=True,
)

model = UNet(configs.channel, configs.embedding_dim)
num_parameters = count_parameters(model)
print("Total number of trainable parameters is: {}".format(num_parameters))


criterion = metric_loss.metricLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=configs.lr, weight_decay=configs.decay_rate
)
scheduler = StepLR(optimizer, step_size=configs.lr_step, gamma=configs.lr_gamma)

start_epoch = 0
if configs.resume:
    if os.path.isfile(configs.model_path):
        print("=> Loading checkpoint '{}'".format(configs.model_path))
        checkpoint = torch.load(configs.model_path)
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        if configs.use_cuda:
            transfer_optimizer_to_gpu(optimizer)
        print(
            "=> Loaded checkpoint {} (epoch {})".format(configs.model_path, start_epoch)
        )
    else:
        print("No checkpoint found at {}".format(configs.model_path))

ckpt_save_path = os.path.join(configs.exp_dir)
if not os.path.exists(ckpt_save_path):
    os.mkdir(ckpt_save_path)

model.to(device)
model.train()

best_loss = np.inf
best_val_loss = np.inf
patience_counter = 0

for epoch_num in range(start_epoch, configs.max_epoch_num):
    print("Learning_rate: :{:.6f}".format(optimizer.param_groups[0]["lr"]))
    start = time.time()

    running_loss = 0

    # Training Step
    progress_bar_train = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch_num + 1}/{configs.max_epoch_num}]")
    model.train()

    for iter_i, batch_data in progress_bar_train:
        images = batch_data["image"].to(device)
        labels = batch_data["label"]

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds.cpu(), labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if iter_i % 10 == 0:
            writer.add_scalar('Loss/Train Step', running_loss, epoch_num)
            progress_bar_train.set_postfix(loss=running_loss / (iter_i + 1))

    scheduler.step()
    num_batches = iter_i + 1
    train_loss = running_loss / num_batches

    print(
        "****** Epoch: [{}/{}], train loss:{:.4f}".format(
             epoch_num + 1, configs.max_epoch_num, train_loss
        )
    )
    writer.add_scalar('Loss/Train Epoch', train_loss, epoch_num)

    # Validation Step
    model.eval()
    val_loss = 0
    progress_bar_val = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation [{epoch_num + 1}/{configs.max_epoch_num}]")
    with torch.no_grad():
        for iter_i, batch_data in progress_bar_val:
            images = batch_data["image"].to(device)
            labels = batch_data["label"]
            preds = model(images)
            loss = criterion(preds.cpu(), labels)
            val_loss += loss.item()

            if iter_i % 10 == 0:
                writer.add_scalar('Loss/Val Step', val_loss, epoch_num)
                progress_bar_val.set_postfix(loss=val_loss / (iter_i + 1))
    
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")
    writer.add_scalar('Loss/Val Epoch', val_loss, epoch_num)

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        print(f"=> Saving checkpoint at epoch {epoch_num + 1} for best val loss")
        print(f"Best val loss: {best_val_loss:.4f}")
        save_checkpoint(
            {
                "epoch": epoch_num + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            is_best=False,
            checkpoint=ckpt_save_path,
            filename="best_val_loss.pth.tar",
        )
    else:
        patience_counter += 1
        print(f"No improvement in validation loss for {patience_counter} epochs.")
        
    if patience_counter >= configs.patience:
        print(f"Early stopping triggered after {patience_counter} epochs with no improvement.")
        break
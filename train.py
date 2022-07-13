import torch
import logging
import argparse
import os
import albumentations as A

from albumentations.pytorch import transforms
from albumentations.augmentations import geometric
from torch import optim, nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from model import UNet
from dataset import BasicDataset
from loss import DiceLoss

ROOT = Path(os.getcwd())
dir_checkpoint = Path('./checkpoints/')


def train(args):
    assert args.num_classes < 2, "The number of class must be at least 2, binary"
    epochs, batch_size, imgsz, lr, val_epoch = args.epochs, args.batch_size, args.imgsz, args.lr, args.val_epoch
    img_dir, mask_dir = os.path.join(args.dataset,'images'), os.path.join(args.dataset,'masks')

    amp, save_checkpoint = True, 1

    model = UNet(in_channels=args.input_channel, n_classes=args.num_classes)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.to(device)

    # Create datasets
    transform = A.Compose([
        geometric.resize.LongestMaxSize(max_size=imgsz),
        transforms.ToTensorV2(),
    ])
    dataset = BasicDataset(img_dir=img_dir, mask_dir=mask_dir, num_classes=args.num_classes, transform=transform)

    # Split up datasets into train/validation set
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Loss
    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes=args.num_classes)

    # Optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=2)

    # Gradient Scaler for Automated Mixed Precision
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Displaly training setting
    logging.info(f'''Starting training:
    Epochs:          {epochs}
    Batch size:      {batch_size}
    Learning rate:   {lr}
    Training size:   {n_train}
    Validation size: {n_val}
    Checkpoints:     {save_checkpoint}
    Device:          {device}
    Mixed Precision: {amp}
    ''')

    # Begin training
    global_step = 0

    for epoch in range(1, epochs+1):
        model.train()
        step, epoch_loss = 0, 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit=' batch')
        for idx, (imgs, masks) in pbar:
            imgs = imgs.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=amp):
                preds = model(imgs)
                loss = criterion(preds,masks) + dice_loss(preds, masks)

                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # pbar.update(imgs.shape[0])
                step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        global_step += step
        epoch_loss = epoch_loss / step
        pbar.set_postfix(**{'loss (batch)': epoch_loss})

        # Evaluation round
        
        if val_epoch > 0 and ((epoch) % val_epoch == 0):
                val_score = validate(model, val_loader, dice_loss, device)
                scheduler.step(val_score)
                logging.info(f'Validation Dice score: {val_score:.3f}')

        if save_checkpoint:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')


def validate(model, dataloader, dice_loss, device):
    model.eval()
    dice = 0
    for idx, (imgs, masks) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Validation round', unit=' batch'):
        imgs = imgs.to(device).float()
        masks = masks.to(device, dtype=torch.long)

        with torch.no_grad():
            preds = model(imgs)
            dice += dice_loss.dice_coeff(preds,masks)
    model.train()
    return dice / len(dataloader)



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'best.pth')
    parser.add_argument('--dataset', type=str, default=ROOT / 'dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--input-channel', type=int, default=3)
    parser.add_argument('--val-epoch', type=int, default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_opt()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(args)
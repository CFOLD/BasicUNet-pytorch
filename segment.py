import torch
import argparse
import logging
import os
import numpy as np
import albumentations as A
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from albumentations.pytorch import transforms
from albumentations.augmentations import geometric
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from model import UNet
from dataset import ImageDataset


ROOT = Path(os.getcwd())


def segment(args):
    imgsz, mask_thres = args.imgsz, args.mask_threshold
    source_dir = str(args.source)

    model = UNet(in_channels=args.input_channel, n_classes=args.num_classes)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.to(device)

    # Create datasets
    transform = A.Compose([
        geometric.resize.LongestMaxSize(max_size=imgsz),
        transforms.ToTensorV2(),
    ])
    dataset = ImageDataset(img_dir=source_dir, transform=transform)

    # Create dataloaders
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)

    # Begin prediction
    model.eval()
    step, epoch_loss = 0, 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Segmentation', unit=' img')
    for idx, (imgs, (h, w, c, name)) in pbar:
        imgs = imgs.to(device, dtype=torch.float32)

        with torch.no_grad():
            preds = F.softmax(model(imgs), dim=1)[0]
            resizeTf = A.Compose([
                geometric.resize.LongestMaxSize(max_size=max([h,w]).item()),
                transforms.ToTensorV2(),
            ])
            mask_tensor = resizeTf(image=np.array(to_pil_image(preds)) )['image']
            mask = F.one_hot(mask_tensor.argmax(dim=0), model.n_classes).numpy()
            mask = mask.astype('uint8') * 255
            mask = np.where(mask_tensor.permute(1,2,0) >= int(mask_thres*255), mask, 0)
            mask = mask[:,:,1] if model.n_classes == 2 else mask
            mask = Image.fromarray(mask)
            mask.save(ROOT / 'runs' / (name[0][:-4]+'.jpg'), "JPEG")

            


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'best.pth')
    parser.add_argument('--source', type=str, default=ROOT / 'images')
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--mask-threshold', type=float, default=0.5)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--input-channel', type=int, default=3)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_opt()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    segment(args)
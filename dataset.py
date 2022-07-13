import glob
import numpy as np
import cv2

from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import transforms
from os.path import basename

class BasicDataset(Dataset):
    def __init__(self, img_dir, mask_dir, num_classes=2, transform=None):
        self.img_list = sorted([file for file in glob.glob(img_dir+'/*') if not basename(file).startswith('.')])
        assert len(self.img_list), 'Check the dataset: make sure the images are in the directory'
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.transform = transform if transform else transforms.ToTensorV2()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        mask_path = self.mask_dir + '/' + basename(img_path)[:-4]+ '_mask.gif'
        # To do: support npy format for Mutli-class exceeding 3-class

        img = cv2.imread(img_path)
        try:
            mask = Image.open(mask_path)
            mask = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR) if mask.mode == 'RGB' else np.array(mask)
        except FileNotFoundError:
            w, h = img.size
            c = self.num_classes
            mask = np.zeros((h, w, c), dtype='uint8')

        transformed = self.transform(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        return img, mask


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_list = sorted([file for file in glob.glob(img_dir+'/*') if not basename(file).startswith('.')])
        assert len(self.img_list), 'Check the dataset: make sure the images are in the directory'
        self.transform = transform if transform else transforms.ToTensorV2()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = cv2.imread(img_path)
        h,w,c = img.shape

        transformed = self.transform(image=img)
        img = transformed['image']

        return img, (h,w,c,basename(img_path))
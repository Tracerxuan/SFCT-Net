from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import cv2


class BUSI_Dataset(Dataset):
    def __init__(self, root_dir, category='benign', transform=None):
        self.transform = transform
        self.data = []
        category_path = os.path.join(root_dir, category)

        for file in os.listdir(category_path):
            if '_mask' not in file and file.endswith('.png'):
                img_name = file
                img_path = os.path.join(category_path, img_name)

                mask_files = [f for f in os.listdir(category_path)
                              if f.startswith(img_name.replace('.png', '')) and '_mask' in f]
                mask_paths = [os.path.join(category_path, f) for f in mask_files]

                self.data.append((img_path, mask_paths))

        print(f'\n[{category}] Read {len(self.data)} image-mask pairs.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_paths = self.data[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        merged_mask = None
        for path in mask_paths:
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if merged_mask is None:
                merged_mask = m
            else:
                merged_mask = np.maximum(merged_mask, m)

        merged_mask = np.where(merged_mask > 0, 1, 0).astype('uint8')

        if self.transform:
            augmented = self.transform(image=img, mask=merged_mask)
            img = augmented['image']
            mask = augmented['mask'].unsqueeze(0).float()

        return img, mask, os.path.basename(img_path)

class STU_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_files = sorted([
            f for f in os.listdir(root_dir) if f.startswith('Test_Image') and f.endswith('.png')
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        num = image_name.split('_')[-1].split('.')[0]
        mask_name = f"mask_{num}.png"

        image_path = os.path.join(self.root_dir, image_name)
        mask_path = os.path.join(self.root_dir, mask_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask > 0, 1, 0).astype('uint8')

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].unsqueeze(0).float()

        return image, mask, os.path.basename(image_path)

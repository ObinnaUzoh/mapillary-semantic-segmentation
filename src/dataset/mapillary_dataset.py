import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A

N_LABELS = 124
TEST_SPLIT_FROM_TRAIN = 0.05
TRAIN_SPLIT_FROM_TRAIN = 1-TEST_SPLIT_FROM_TRAIN

def make_augmentations(image_height, image_width, random_crop_min=0.6, p=0.5):
    random_crop_ratio = random_crop_min + random.random()*(1-random_crop_min)
    crop_height = int(random_crop_ratio*image_height)
    crop_width = int(random_crop_ratio*image_width)

    weak_transform = A.Compose([
        A.RandomBrightnessContrast(p=p),
        A.CLAHE(p=p),
        A.Blur(blur_limit=3, p=p),
        A.HueSaturationValue(p=p),
    ])

    strong_transform = A.Compose([
        A.RandomCrop(width=crop_width, height=crop_height),
        A.Resize(width=image_width, height=image_height),
        A.HorizontalFlip(p=p),
        A.OpticalDistortion(p=p),
        A.ShiftScaleRotate(p=p),
        A.RandomRotate90(p=p),
        A.Transpose(p=p),
        A.Perspective(p=p),
    ])

    return weak_transform, strong_transform

class Mapillary_SemSeg_Dataset(Dataset):
    def __init__(self, split, data_dir, image_shape, augment=False, version='v2.0', verbose=False):
        self.data_dir = data_dir
        self.version = version
        if split == 'testing':
            self.split = 'training'
        else:
            self.split = split 
        self.image_height, self.image_width = image_shape
        self.image_filenames = os.listdir(os.path.join(self.data_dir, f'{self.split}/images'))
        if split=='testing' or split=='training':
            random.seed(0)
            self.image_filenames.sort()
            random.shuffle(self.image_filenames)
            if split=='training':
                self.image_filenames = self.image_filenames[:int(len(self.image_filenames)*TRAIN_SPLIT_FROM_TRAIN)]
            else:
                self.image_filenames = self.image_filenames[int(len(self.image_filenames)*TRAIN_SPLIT_FROM_TRAIN):]
        self.augment = augment
        self.verbose = verbose
        if self.augment:
            self.weak_transform, self.strong_transform = make_augmentations(self.image_height, self.image_width)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_id = os.path.splitext(image_filename)[0]
        image_path = os.path.join(self.data_dir, f"{self.split}/images/{image_id}.jpg")
        label_path = os.path.join(self.data_dir, f"{self.split}/{self.version}/labels/{image_id}.png")
        
        image = np.array(Image.open(image_path).resize((self.image_width, self.image_height)))
        label = np.array(Image.open(label_path).resize((self.image_width, self.image_height), Image.NEAREST))

        mask = np.zeros((*label.shape, N_LABELS))

        for j in range(N_LABELS):
            mask[:, :, j][label == j] = 1

        if self.augment:
            augment_mode = random.randint(0, 3)
            if self.verbose:
                print(f'augment_mode: {augment_mode}')
            if augment_mode==0:
                transformed = {'image': image, 'mask': mask}
            if augment_mode==1:
                transformed = self.weak_transform(image=image, mask=mask)
            elif augment_mode==2:
                transformed = self.strong_transform(image=image, mask=mask)
            elif augment_mode==3:
                transformed = self.weak_transform(image=image, mask=mask)

                transformed = self.strong_transform(image=transformed['image'], mask=transformed['mask'])
            
            image = transformed['image']
            mask = transformed['mask']
        
        return image.transpose(2, 0, 1).astype(np.float32)/255, mask.transpose(2, 0, 1).astype(np.float32)


if __name__ == '__main__':
    train_dataset = Mapillary_SemSeg_Dataset('training', '../../data', (512, 512), augment=True)
    val_dataset = Mapillary_SemSeg_Dataset('validation', '../../data', (512, 512))
    # test_dataset = Mapillary_SemSeg_Dataset('testing', '../../data')

    img, label = train_dataset[0]

    print(img.shape, label.shape)

    
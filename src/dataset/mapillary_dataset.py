import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

N_LABELS = 124

class Mapillary_SemSeg_Dataset(Dataset):
    def __init__(self, split, data_dir, version='v2.0'):
        self.data_dir = data_dir
        self.version = version
        self.split = split
        self.image_filenames = os.listdir(os.path.join(self.data_dir, f'{split}/images'))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_id = os.path.splitext(image_filename)[0]
        image_path = os.path.join(self.data_dir, f"{self.split}/images/{image_id}.jpg")
        label_path = os.path.join(self.data_dir, f"{self.split}/{self.version}/labels/{image_id}.png")
        
        image = np.array(Image.open(image_path))
        label = np.array(Image.open(label_path))

        channel_labels = np.zeros((N_LABELS, *label.shape))

        for j in range(N_LABELS):
            channel_labels[j][label == j] = 1

        return image.transpose(2, 0, 1), channel_labels


if __name__ == '__main__':
    train_dataset = Mapillary_SemSeg_Dataset('training', '../../data')
    val_dataset = Mapillary_SemSeg_Dataset('validation', '../../data')
    # test_dataset = Mapillary_SemSeg_Dataset('testing', '../../data')

    img, label = train_dataset[0]

    print(img.shape, label.shape)

    
import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class TrainDataset(Dataset):
    def __init__(self, data_frames, root_dir, transform=None):
        df = pd.read_csv(data_frames)
        self.data = df[df['video_idx'].isin(np.arange(0, 40))]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = f"{self.root_dir}/{self.data.iloc[idx]['image_path']}"
        image = Image.open(path)
        image_mat = np.array(image)
        if self.transform is not None:
            image_mat = self.transform(image=image_mat)['image']
        return image_mat

if __name__ == '__main__':
    augs = Compose(
        [
            Resize(height=224, width=224),
            Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2(),
        ]
    )
    train_dataset = TrainDataset('../../dataframes_cataract-101/cataract_split_250px_5fps.csv', '../../images_cataract_101', augs)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs in tqdm(train_dataloader):
        psum += inputs.sum(axis=[0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])

    # pixel count
    count = len(train_dataset) * 224 * 224

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # output
    print("mean: " + str(total_mean))
    print("std:  " + str(total_std))
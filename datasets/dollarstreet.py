import ast
import os

import pandas as pd
import torch

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

'''
The Dollar Street dataset is a dataset of images of people's homes from around the world (https://www.gapminder.org/dollar-street).
Data must be downloaded from Kaggle: https://www.kaggle.com/datasets/mlcommons/the-dollar-street-dataset

kaggle datasets download -d mlcommons/the-dollar-street-dataset -p <root_dir>
unzip <root_dir>/the-dollar-street-dataset.zip -d <root_dir>
'''

class DollarStreetDataset(Dataset):
    def __init__(self, csv_file_path, root_dir, pre_filter=None, transform=None):
        self.data = pd.read_csv(csv_file_path)
        self.root_dir = root_dir

        # Default transform
        if not transform:
            transform = transforms.Compose([
                transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalization based on train split (see get_normalization_mean_std below)
            ])
        self.transform = transform

        # Apply pre_filter
        if pre_filter:
            self.data = self.data[self.data.apply(pre_filter, axis=1)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image and label
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 5]) # 5 is the column imageRelPath
        image = Image.open(img_name).convert("RGB")
        label = ast.literal_eval(self.data.iloc[idx, 10])[0] # 10 is the column imagenet_sysnet_id, taking the first label (alternatively we could also perform multi-class classification by taking all labels)

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label
    

def get_dollarstreet(root_dir, split, batch_size, shuffle, num_workers, pre_filter=None):
    assert split in ["train", "test"], "Split must be either 'train' or 'test'"
    csv_name = f"images_v2_imagenet_{split}.csv"

    dataset = DollarStreetDataset(
        csv_file=os.path.join(root_dir, csv_name),
        root_dir=root_dir,
        pre_filter=pre_filter
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader


def get_normalization_mean_std(root_dir):
    transform = transforms.Compose([
        transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    csv_name = f"images_v2_imagenet_train.csv"

    data_train = DollarStreetDataset(csv_file=os.path.join(root_dir, csv_name), root_dir=root_dir, transform=transform)
    dataloader = DataLoader(data_train, batch_size=64, shuffle=False, num_workers=4)

    def calculate_mean_std(dataloader):
        mean = 0
        std = 0
        total_images_count = 0
        
        for images, _ in dataloader:
            batch_samples = images.size(0) # Batch size (the last batch can have smaller size)
            images = images.view(batch_samples, images.size(1), -1) # Reshape images to (batch_size, channels, width*height)
            mean += images.mean(2).sum(0) # Sum up means for each channel
            std += images.std(2).sum(0) # Sum up std for each channel
            total_images_count += batch_samples

        mean /= total_images_count
        std /= total_images_count

        return mean, std

    # Calculate mean and std
    return calculate_mean_std(dataloader)

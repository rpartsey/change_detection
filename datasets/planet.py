import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from utils.image import read_tif


class PlanetClassificationDataset(Dataset):
    """Planet dataset."""

    def __init__(self, csv_file, **kwargs):
        """
        Args:
            csv_file (string): Path to the csv file with data locations.
        """
        self.df = pd.read_csv(csv_file)
        self.image_transforms = kwargs.get('image_transforms')
        self.label_transforms = kwargs.get('label_transforms')
        self.augmentations = kwargs.get('augmentations')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row['image_path']
        image = read_tif(image_path)
        label = np.array([row['label']])

        if self.augmentations:
            image = self.augmentations(image)

        if self.image_transforms:
            image = self.image_transforms(image)

        if self.label_transforms:
            label = self.label_transforms(label)

        meta = {}  # meta data

        return image, label, meta

    def __len__(self):
        return len(self.df)

    @classmethod
    def from_config(cls, config):
        return cls(
            csv_file=config.csv_path,
            image_transforms=config.image_transforms,
            label_transforms=config.label_transforms,
            augmentations=config.augmentations
        )


class PlanetClassificationTestDataset:
    pass


class DataLoader(TorchDataLoader):
    @classmethod
    def from_config(cls, dataset, config):
        return cls(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            sampler=config.sampler,
            num_workers=config.num_workers
        )

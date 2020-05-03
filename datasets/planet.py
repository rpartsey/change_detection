import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from utils.image import read_tif


class PlanetClassificationDataset(Dataset):
    """Planet classification dataset."""

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


class PlanetClassificationDatasetV2(PlanetClassificationDataset):
    """
    Planet dataset to be used with CropNonEmptyMaskIfExists augmentation.
    """
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row['image_path']
        mask_path = image_path.replace('images', 'masks')

        image = read_tif(image_path).transpose((1, 2, 0))
        mask = read_tif(mask_path).transpose((1, 2, 0))
        label = np.array([row['label']])

        if self.augmentations:
            image, mask = self.augmentations(image, mask)

        if self.image_transforms:
            image = self.image_transforms(image)

        if self.label_transforms:
            label = self.label_transforms(label)

        meta = {}  # meta data

        return image, label, meta


class PlanetSegmentationDataset(Dataset):
    """Planet segmentation dataset."""

    def __init__(self, csv_file, **kwargs):
        """
        Args:
            csv_file (string): Path to the csv file with data locations.
        """
        self.df = pd.read_csv(csv_file)
        self.image_transforms = kwargs.get('image_transforms')
        self.mask_transforms = kwargs.get('mask_transforms')
        self.augmentations = kwargs.get('augmentations')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = read_tif(row['image_path']).transpose((1, 2, 0))
        mask = read_tif(row['mask_path']).transpose((1, 2, 0))

        if self.augmentations:
            image, mask = self.augmentations(image, mask)

        if self.image_transforms:
            image = self.image_transforms(image)

        if self.mask_transforms:
            mask = self.mask_transforms(mask)

        meta = {}  # meta data

        return image, mask#, meta

    def __len__(self):
        return len(self.df)

    @classmethod
    def from_config(cls, config):
        return cls(
            csv_file=config.csv_path,
            image_transforms=config.image_transforms,
            mask_transforms=config.mask_transforms,
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

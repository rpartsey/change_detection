import numpy as np
import pandas as pd
from torch.utils.data import Dataset

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


class PlanetClassificationTestDataset:
    pass

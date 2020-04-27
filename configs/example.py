"""
Basic example configuration file.

Experiment description ...
mean=[4693.149574344914, 4083.8567912125004, 3253.389157030059, 4042.120897153529],
std=[533.0050173177232, 532.784091756862, 574.671063551312, 913.357907430358]
"""
from torch import nn, optim

import transforms as t
from augs import StandardAugmentation
from models.resnet34 import get_resnet34


class ExperimentConfig:
    directory = 'path/to/directory'
    device = 'cuda:1'
    num_epochs = 200
    random_state = 2412
    model = get_resnet34(in_planes=4, classes=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


class DatasetConfig:
    augmentations = StandardAugmentation()
    image_transforms = t.Compose([
        # t.RGBOnly(),
        t.ToNumpyInt32(),
        t.FromNumpy(),
        t.ToTorchFloat(),
        t.Normalize(
            mean=[4693.149574344914, 4083.8567912125004, 3253.389157030059, 4042.120897153529],
            std=[533.0050173177232, 532.784091756862, 574.671063551312, 913.357907430358]
        ),
    ])
    label_transforms = t.Compose([
        t.FromNumpy(),
        t.ToTorchFloat(),
    ])


class TrainDatasetConfig(DatasetConfig):
    csv_path = '/datasets/rpartsey/satellite/planet/csv/classification/train.csv'


class ValidationDatasetConfig(DatasetConfig):
    csv_path = '/datasets/rpartsey/satellite/planet/csv/classification/val.csv'


class TrainDataloaderConfig: # noqa
    batch_size = 32
    shuffle = True
    sampler = None
    num_workers = 4


class ValDataloaderConfig: # noqa
    batch_size = 32
    shuffle = False
    sampler = None
    num_workers = 4

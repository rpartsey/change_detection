"""
Basic example configuration file.

Experiment description ...
mean=[4693.149574344914, 4083.8567912125004, 3253.389157030059, 4042.120897153529],
std=[533.0050173177232, 532.784091756862, 574.671063551312, 913.357907430358]
"""
from torch import nn, optim
from torchvision import models

import transforms as t
from augs import StandardAugmentation
from models.resnet34 import get_resnet34


class ExperimentConfig:
    directory = 'test3'
    device = 'cuda:1'
    save_each_epoch = False
    num_epochs = 200
    random_state = 2412
    model = get_resnet34(in_planes=4, classes=1)

    # model = models.resnet18(pretrained=True)
    # num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model.fc = nn.Linear(num_ftrs, 1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)


class DatasetConfig:
    augmentations = StandardAugmentation()
    image_transforms = t.Compose([
        # t.RGBOnly(),
        t.ToNumpyInt32(),
        t.FromNumpy(),
        t.ToTorchFloat(),
        # t.Normalize(
        #     mean=[4693.149574344914, 4083.8567912125004, 3253.389157030059],
        #     std=[533.0050173177232, 532.784091756862, 574.671063551312]
        # ),
        t.Normalize(
            mean=[4693.149574344914, 4083.8567912125004, 3253.389157030059, 4042.120897153529],
            std=[533.0050173177232, 532.784091756862, 574.671063551312, 913.357907430358]
        )
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
    batch_size = 8
    shuffle = True
    sampler = None
    num_workers = 4


class ValDataloaderConfig: # noqa
    batch_size = 8
    shuffle = False
    sampler = None
    num_workers = 4

"""
Basic example configuration file.

Experiment description ...
mean=[4693.149574344914, 4083.8567912125004, 3253.389157030059, 4042.120897153529],
std=[533.0050173177232, 532.784091756862, 574.671063551312, 913.357907430358]
"""
from torch import optim

import transforms as t
from augs import SmartCrop, CenterCrop
import segmentation_models_pytorch as smp


class ExperimentConfig:
    directory = 'ba_segmentation'
    device = 'cuda:0'
    save_each_epoch = False
    num_epochs = 200
    random_state = 2412

    model = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=4, classes=1, activation='sigmoid')
    criterion = smp.utils.losses.DiceLoss() # smp.utils.losses.BCELoss() + smp.utils.losses.DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    metrics = [smp.utils.metrics.IoU(threshold=0.5), ]


class DatasetConfig:
    mask_transforms = t.Compose([
        t.ChannelsFirst(),
        t.FromNumpy(),
        t.ToTorchFloat(),
    ])


class TrainDatasetConfig(DatasetConfig):
    augmentations = SmartCrop(256, 256, p=1.0)
    image_transforms = t.Compose([
        # t.RGBOnly(),
        t.ChannelsFirst(),
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
    csv_path = '/datasets/rpartsey/satellite/planet/smart_crop/train.csv'


class ValidationDatasetConfig(DatasetConfig):
    augmentations = CenterCrop(256, 256, p=1.0)
    image_transforms = t.Compose([
        # t.RGBOnly(),
        t.ChannelsFirst(),
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
    csv_path = '/datasets/rpartsey/satellite/planet/smart_crop/val.csv'


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

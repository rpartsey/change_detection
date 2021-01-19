import os
from torch import optim

import transforms as t
from augs import SmartCrop, CenterCrop
import segmentation_models_pytorch as smp
from utils.general import set_random_seed

set_random_seed(2412)


class ExperimentConfig:
    directory = 'segmentation/pre-event/unet/baseline'
    device = 'cuda:0'
    num_epochs = 55+1
    random_state = 2412

    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=8, classes=1, activation='sigmoid')
    criterion = smp.utils.losses.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    metrics = [smp.utils.metrics.IoU(threshold=0.5), ]


class DatasetConfig:
    mask_transforms = t.Compose([
        t.ChannelsFirst(),
        t.FromNumpy(),
        t.ToTorchFloat(),
    ])
    base_csv_dir = '/datasets/rpartsey/satellite/planet/planet_dataset/train-val/random_split'


class TrainDatasetConfig(DatasetConfig):
    augmentations = SmartCrop(256, 256, p=1.0)
    image_transforms = t.Compose([
        t.ChannelsFirst(),
        t.ToNumpyInt32(),
        t.FromNumpy(),
        t.ToTorchFloat(),
        t.Normalize(
            mean=[4417.258621276464, 3835.2537312971936, 3065.427994856266, 3783.5501700000373,
                  4417.258621276464, 3835.2537312971936, 3065.427994856266, 3783.5501700000373],
            std=[805.3352649209319, 752.9507977334065, 769.0657720493105, 1136.0581964787941,
                 805.3352649209319, 752.9507977334065, 769.0657720493105, 1136.0581964787941]
        )
    ])
    csv_path = os.path.join(DatasetConfig.base_csv_dir, 'train.csv')


class ValidationDatasetConfig(DatasetConfig):
    augmentations = CenterCrop(256, 256, p=1.0)
    image_transforms = t.Compose([
        t.ChannelsFirst(),
        t.ToNumpyInt32(),
        t.FromNumpy(),
        t.ToTorchFloat(),
        t.Normalize(
            mean=[4417.258621276464, 3835.2537312971936, 3065.427994856266, 3783.5501700000373,
                  4417.258621276464, 3835.2537312971936, 3065.427994856266, 3783.5501700000373],
            std=[805.3352649209319, 752.9507977334065, 769.0657720493105, 1136.0581964787941,
                 805.3352649209319, 752.9507977334065, 769.0657720493105, 1136.0581964787941]
        )
    ])
    csv_path = os.path.join(DatasetConfig.base_csv_dir, 'val.csv')


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

"""
Basic example configuration file.

Experiment description ...
mean=[4693.149574344914, 4083.8567912125004, 3253.389157030059, 4042.120897153529],
std=[533.0050173177232, 532.784091756862, 574.671063551312, 913.357907430358]
"""
import transforms as t
from augs import StandardAugmentation


class ExperimentConfig:
    directory = 'path/to/directory'
    device = 'cuda:0'
    num_epochs = 30


class ModelConfig:
    pass


class DatasetConfig:
    train_csv_path = '/datasets/rpartsey/satellite/planet/csv/classification/train.csv'
    val_csv_path = '/datasets/rpartsey/satellite/planet/csv/classification/val.csv'
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
    # label_transforms = None
    label_transforms = t.Compose([
        t.FromNumpy(),
        t.ToTorchFloat(),
    ])


class TrainDataloaderConfig: # noqa
    batch_size = 8
    shuffle = True
    sampler = None
    num_workers = 4


class ValDataloaderConfig: # noqa
    batch_size = 2
    shuffle = False
    sampler = None
    num_workers = 4

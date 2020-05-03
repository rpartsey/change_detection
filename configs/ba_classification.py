"""
Basic example configuration file.

Experiment description ...
mean=[4693.149574344914, 4083.8567912125004, 3253.389157030059, 4042.120897153529],
std=[533.0050173177232, 532.784091756862, 574.671063551312, 913.357907430358]
"""
import torch
from torch import nn, optim
from torchvision import models
from torch.functional import F
import segmentation_models_pytorch as smp

import transforms as t
from augs import SmartCrop, CenterCrop


class BinaryCrossEntropy(nn.Module):
    def forward(self, logit, truth):
        logit = logit.view(-1)
        truth = truth.view(-1)
        assert(logit.shape==truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

        return loss.mean()

class ExperimentConfig:
    directory = 'ba_classification'
    device = 'cuda:0'
    save_each_epoch = False
    num_epochs = 200
    random_state = 2412

    model = models.resnet34(pretrained=True)
    conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    weight = model.conv1.weight.clone()

    with torch.no_grad():
        conv.weight[:, :3] = weight
        conv.weight[:, 3] = weight[:, 0]

    model.conv1 = conv
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    criterion = BinaryCrossEntropy()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)


class DatasetConfig:
    label_transforms = t.Compose([
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

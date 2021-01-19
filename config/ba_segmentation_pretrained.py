"""
Basic example configuration file.

Experiment description ...
mean=[4693.149574344914, 4083.8567912125004, 3253.389157030059, 4042.120897153529],
std=[533.0050173177232, 532.784091756862, 574.671063551312, 913.357907430358]
"""
import torch
from torch import optim, nn
from torch.functional import F
from segmentation_models_pytorch.utils.losses import base

import transforms as t
from augs import SmartCrop, CenterCrop, SmartCropColorAndScale
import segmentation_models_pytorch as smp


class FocalLoss(base.Loss):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True,  **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class ExperimentConfig:
    directory = 'segmentation/unet/pretrained_bce_dice_smoke'
    device = 'cuda:0'
    save_each_epoch = False
    num_epochs = 200
    random_state = 2412

    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=4, classes=1, activation='sigmoid')
    criterion = smp.utils.losses.BCELoss() + smp.utils.losses.DiceLoss()
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
            mean=[4417.258621276464, 3835.2537312971936, 3065.427994856266, 3783.5501700000373],
            std=[805.3352649209319, 752.9507977334065, 769.0657720493105, 1136.0581964787941]
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
            mean=[4417.258621276464, 3835.2537312971936, 3065.427994856266, 3783.5501700000373],
            std=[805.3352649209319, 752.9507977334065, 769.0657720493105, 1136.0581964787941]
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

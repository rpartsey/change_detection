import os
import albumentations as albu
from torch import nn, optim

import segmentation_models_pytorch as smp


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


class ExperimentConfig:
    directory = 'camvid_segmentation'
    device = 'cuda:0'
    save_each_epoch = False
    num_epochs = 200
    random_state = 2412

    encoder = 'se_resnext50_32x4d'
    encoder_weights = 'imagenet'
    classes = ['car']
    activation = 'sigmoid'

    model = smp.FPN(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=len(classes),
        activation=activation,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    metrics = [smp.utils.metrics.IoU(threshold=0.5), ]
    criterion = smp.utils.losses.DiceLoss()
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)


class DatasetConfig:
    data_dir = '/datasets/rpartsey/SegNet-Tutorial/CamVid'


class TrainDatasetConfig(DatasetConfig):
    images_dir = os.path.join(DatasetConfig.data_dir, 'train')
    masks_dir = os.path.join(DatasetConfig.data_dir, 'trainannot')
    augmentations = get_training_augmentation()
    preprocessing = get_preprocessing(ExperimentConfig.preprocessing_fn)
    classes = ExperimentConfig.classes


class ValidationDatasetConfig(DatasetConfig):
    images_dir = os.path.join(DatasetConfig.data_dir, 'val')
    masks_dir = os.path.join(DatasetConfig.data_dir, 'valannot')
    augmentations = get_validation_augmentation()
    preprocessing = get_preprocessing(ExperimentConfig.preprocessing_fn)
    classes = ExperimentConfig.classes


class TrainDataloaderConfig: # noqa
    batch_size = 8
    shuffle = True
    sampler = None
    num_workers = 1


class ValDataloaderConfig: # noqa
    batch_size = 8
    shuffle = False
    sampler = None
    num_workers = 1

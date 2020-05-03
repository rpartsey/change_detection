"""
Ants vs bees classification experiment.

This code is written purely for testing purposes.
"""

import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets, models
from experiments.binary_classification import train_epoch, calculate_metrics, print_metrics, write_metrics
from models.resnet34 import get_resnet34
from utils.general import set_random_seed, create_experiment_log_dir
import transforms as t


def run_experiment():
    set_random_seed(2412)
    logdir_path = create_experiment_log_dir('ants_and_bees')
    writer = SummaryWriter(logdir_path)

    data_transforms = {
        'image': {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        },
        'label': transforms.Compose([
            lambda label: np.array([label]),
            t.FromNumpy(),
            t.ToTorchFloat(),
        ])
    }

    data_dir = '/datasets/rpartsey/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms['image'][x],
                                              data_transforms['label'])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = get_resnet34(in_planes=3, classes=1)
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, 1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    num_epochs = 25
    for epoch in range(num_epochs):
        print('\nEpoch {}'.format(epoch))
        train_loss, train_metrics = train_epoch(model, criterion, optimizer, dataloaders['train'], device, exp_lr_scheduler)
        train_metrics.update(phase='Train', loss=train_loss)
        print_metrics(train_metrics)
        write_metrics(writer, train_metrics, epoch)

        val_loss, val_metrics = calculate_metrics(model, criterion, dataloaders['val'], device)
        val_metrics.update(phase='Validation', loss=val_loss)
        print_metrics(val_metrics)
        write_metrics(writer, val_metrics, epoch)


run_experiment()

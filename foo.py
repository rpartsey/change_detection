import os

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from datasets.planet import PlanetSegmentationDatasetV2, DataLoader
from experiments.binary_segmentation import train_epoch, print_metrics, write_metrics, calculate_metrics
from utils.general import create_experiment_log_dir


def run_experiment(config):
    logdir_path = create_experiment_log_dir(config.ExperimentConfig.directory)
    writer = SummaryWriter(logdir_path)

    train_dataset = PlanetSegmentationDatasetV2.from_config(config.TrainDatasetConfig)
    val_dataset = PlanetSegmentationDatasetV2.from_config(config.ValidationDatasetConfig)

    train_loader = DataLoader.from_config(train_dataset, config.TrainDataloaderConfig)
    val_loader = DataLoader.from_config(val_dataset, config.ValDataloaderConfig)

    device = config.ExperimentConfig.device
    model = config.ExperimentConfig.model
    model.to(device)

    criterion = config.ExperimentConfig.criterion
    optimizer = config.ExperimentConfig.optimizer
    metrics = config.ExperimentConfig.metrics
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    # training & validation
    best_iou = 0
    num_epochs = config.ExperimentConfig.num_epochs
    for epoch in range(num_epochs):
        print('\nEpoch {}'.format(epoch))
        train_loss, train_metrics = train_epoch(model, criterion, optimizer, train_loader, metrics, device)
        train_metrics.update(phase='Train', loss=train_loss)
        print_metrics(train_metrics)
        write_metrics(writer, train_metrics, epoch)

        val_loss, val_metrics = calculate_metrics(model, criterion, val_loader, metrics, device)
        val_metrics.update(phase='Validation', loss=val_loss)
        print_metrics(val_metrics)
        write_metrics(writer, val_metrics, epoch)

        if scheduler:
            scheduler.step(val_metrics['loss'])

        if val_metrics['iou_score'] > best_iou:
            best_iou = val_metrics['iou_score']
            torch.save(model, os.path.join(logdir_path, 'best_model.pth'))
            print('Best IoU updated!')

    print(f"\n{' ' * 10}{'*' * 10} Best IoU: {best_iou} {' ' * 10}{'*' * 10}")
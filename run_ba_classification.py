"""
Burned areas classification experiment.
"""
import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import config.ba_classification_efficientnet as experiment_config
from datasets.datasets import PlanetClassificationDatasetV2, DataLoader
from experiments.binary_classification import train_epoch, calculate_metrics, print_metrics, write_metrics
from utils.general import create_experiment_log_dir


def run_experiment(config):
    logdir_path = create_experiment_log_dir(config.ExperimentConfig.directory)
    writer = SummaryWriter(logdir_path)

    train_dataset = PlanetClassificationDatasetV2.from_config(config.TrainDatasetConfig)
    val_dataset = PlanetClassificationDatasetV2.from_config(config.ValidationDatasetConfig)

    train_loader = DataLoader.from_config(train_dataset, config.TrainDataloaderConfig)
    val_loader = DataLoader.from_config(val_dataset, config.ValDataloaderConfig)

    device = config.ExperimentConfig.device
    model = config.ExperimentConfig.model
    print(device)
    model.to(device)

    criterion = config.ExperimentConfig.criterion
    optimizer = config.ExperimentConfig.optimizer
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=8, verbose=True, threshold=1e-3) # ReduceLROnPlateau(optimizer, 'min', patience=8, verbose=True, threshold=1e-4)

    # training & validation
    best_acc = 0
    num_epochs = config.ExperimentConfig.num_epochs
    for epoch in range(num_epochs):
        print('\nEpoch {}'.format(epoch))
        train_loss, train_metrics = train_epoch(model, criterion, optimizer, train_loader, device)
        train_metrics.update(phase='Train', loss=train_loss)
        print_metrics(train_metrics)
        write_metrics(writer, train_metrics, epoch)

        val_loss, val_metrics = calculate_metrics(model, criterion, val_loader, device)
        val_metrics.update(phase='Validation', loss=val_loss)
        print_metrics(val_metrics)
        write_metrics(writer, val_metrics, epoch)

        if scheduler:
            scheduler.step(val_metrics['loss'])

        if best_acc < val_metrics['accuracy']:
            state_dict = {'model': model.state_dict()}
            path = os.path.join(logdir_path, 'best.h5')
            torch.save(state_dict, path)

            best_acc = val_metrics['accuracy']
            print('Best accuracy updated!')


run_experiment(experiment_config)

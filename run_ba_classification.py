"""
Burned areas classification experiment.
"""

import configs.example as experiment_config
from datasets.planet import PlanetClassificationDataset, DataLoader
from experiments.ba_classification import train_epoch, calculate_metrics, print_metrics
from utils.general import set_random_seed


def run_experiment(config):
    set_random_seed(config.ExperimentConfig.random_state)

    train_dataset = PlanetClassificationDataset.from_config(config.TrainDatasetConfig)
    val_dataset = PlanetClassificationDataset.from_config(config.ValidationDatasetConfig)

    train_loader = DataLoader.from_config(train_dataset, config.TrainDataloaderConfig)
    val_loader = DataLoader.from_config(val_dataset, config.ValDataloaderConfig)

    device = config.ExperimentConfig.device
    model = config.ExperimentConfig.model
    model.to(device)

    criterion = config.ExperimentConfig.criterion
    optimizer = config.ExperimentConfig.optimizer

    # training & validation
    num_epochs = config.ExperimentConfig.num_epochs
    for epoch in range(num_epochs):
        print('\nEpoch {}'.format(epoch))
        train_loss, train_metrics = train_epoch(model, criterion, optimizer, train_loader, device)
        train_metrics.update(key='Train', loss=train_loss)
        print_metrics(train_metrics)

        val_loss, val_metrics = calculate_metrics(model, criterion, val_loader, device)
        val_metrics.update(key='Validation', loss=val_loss)
        print_metrics(val_metrics)


run_experiment(experiment_config)

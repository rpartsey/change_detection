"""
Burned areas segmentation experiment.
"""

from torch.utils.tensorboard import SummaryWriter
import configs.ba_segmentation as experiment_config
from datasets.planet import PlanetSegmentationDataset, DataLoader
from experiments.binary_segmentation import train_epoch, calculate_metrics, print_metrics, write_metrics
from utils.general import set_random_seed, create_experiment_log_dir


def run_experiment(config):
    set_random_seed(config.ExperimentConfig.random_state)
    logdir_path = create_experiment_log_dir(config.ExperimentConfig.directory)
    writer = SummaryWriter(logdir_path)

    train_dataset = PlanetSegmentationDataset.from_config(config.TrainDatasetConfig)
    val_dataset = PlanetSegmentationDataset.from_config(config.ValidationDatasetConfig)

    train_loader = DataLoader.from_config(train_dataset, config.TrainDataloaderConfig)
    val_loader = DataLoader.from_config(val_dataset, config.ValDataloaderConfig)

    device = config.ExperimentConfig.device
    model = config.ExperimentConfig.model
    model.to(device)

    criterion = config.ExperimentConfig.criterion
    optimizer = config.ExperimentConfig.optimizer
    metrics = config.ExperimentConfig.metrics

    # training & validation
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


run_experiment(experiment_config)

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from models.vgg16 import get_vgg16
from models.resnet34 import get_resnet34
import torch.optim as optim  # noqa
from tqdm import tqdm

import configs.example as config
from datasets.planet import PlanetClassificationDataset
from metrics import BinaryClassificationMeter


def calculate_metrics(model, criterion, loader, device, threshold=0.5):
    model.eval()

    running_loss = torch.tensor(0.0, dtype=torch.double)
    meter = BinaryClassificationMeter(device)

    with tqdm(loader, desc='Calculating metrics...') as tqdm_loader:  # noqa
        for i, (images, labels, meta) in enumerate(tqdm_loader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = torch.sigmoid(outputs) >= threshold  # noqa

            running_loss += loss.item() * images.size(0)

            preds = preds.type(torch.uint8)  # noqa
            labels = labels.type(torch.uint8)
            meter.update(labels, preds)

    epoch_loss = running_loss / meter.total_count
    metrics = meter.get_metrics()

    return epoch_loss, metrics


def train_epoch(model, criterion, optimizer, loader, device, threshold=0.5):
    model.train()

    running_loss = torch.tensor(0.0, dtype=torch.double)
    meter = BinaryClassificationMeter(device)

    with tqdm(loader, desc='Training...') as tqdm_loader: # noqa
        for i, (images, labels, meta) in enumerate(tqdm_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images) # noqa
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            batch_loss = loss.item() * images.size(0)
            running_loss += batch_loss

            preds = (torch.sigmoid(outputs) >= threshold).type(torch.uint8)  # noqa
            labels = labels.type(torch.uint8)
            meter.update(labels, preds)

    epoch_loss = running_loss / meter.total_count
    metrics = meter.get_metrics()

    return epoch_loss, metrics


train_dataset = PlanetClassificationDataset(
    csv_file=config.DatasetConfig.train_csv_path,
    image_transforms=config.DatasetConfig.image_transforms,
    label_transforms=config.DatasetConfig.label_transforms,
    augmentations=config.DatasetConfig.augmentations
)

val_dataset = PlanetClassificationDataset(
    csv_file=config.DatasetConfig.val_csv_path,
    image_transforms=config.DatasetConfig.image_transforms,
    label_transforms=config.DatasetConfig.label_transforms,
    augmentations=config.DatasetConfig.augmentations
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.TrainDataloaderConfig.batch_size,
    shuffle=config.TrainDataloaderConfig.shuffle,
    sampler=config.TrainDataloaderConfig.sampler,
    num_workers=config.TrainDataloaderConfig.num_workers
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=config.ValDataloaderConfig.batch_size,
    shuffle=config.ValDataloaderConfig.shuffle,
    sampler=config.ValDataloaderConfig.sampler,
    num_workers=config.ValDataloaderConfig.num_workers
)

device = config.ExperimentConfig.device
model = get_resnet34(in_planes=4, classes=1)

# model = get_vgg16(pretrained=True, num_classes=2)
# for param in model.features.parameters():
#     param.requires_grad = False
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# training & validation
num_epochs = config.ExperimentConfig.num_epochs
for epoch in range(num_epochs):
    print('\nEpoch {}'.format(epoch))
    train_loss, train_metrics = train_epoch(model, criterion, optimizer, train_loader, device)
    train_metrics.update(train_loss=train_loss)
    print(
        'Train metrics:\n'
        'Loss: {train_loss}\n'
        'Confusion matrix:\n{conf_mat}\n'
        'Accuracy: {accuracy}\n'
        'Precision: {precision}\n'
        'Recall: {recall}\n'
        'F1: {f1}\n\n'.format(**train_metrics)
    )
    val_loss, val_metrics = calculate_metrics(model, criterion, val_loader, device)
    val_metrics.update(val_loss=val_loss)
    print(
        'Validation metrics:\n'
        'Loss: {val_loss}\n'
        'Confusion matrix:\n{conf_mat}\n'
        'Accuracy: {accuracy}\n'
        'Precision: {precision}\n'
        'Recall: {recall}\n'
        'F1: {f1}\n\n'.format(**val_metrics)
    )

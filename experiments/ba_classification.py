import torch
from tqdm import tqdm

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


def print_metrics(metrics):
    print(
        '{key} metrics:\n'
        'Loss: {loss}\n'
        'Confusion matrix:\n{conf_mat}\n'
        'Accuracy: {accuracy}\n'
        'Precision: {precision}\n'
        'Recall: {recall}\n'
        'F1: {f1}\n\n'.format(**metrics)
    )

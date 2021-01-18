import torch
from tqdm import tqdm

from metrics import AverageMetricsMeter


def calculate_metrics(model, criterion, loader, metrics, device, threshold=0.5):
    model.eval()

    total_count = 0
    running_loss = torch.tensor(0.0, dtype=torch.double)
    meter = AverageMetricsMeter(metrics, device)

    with tqdm(loader, desc='Calculating metrics...') as tqdm_loader:  # noqa
        for i, batch in enumerate(tqdm_loader):
            images, masks = batch[:2]

            images = images.to(device)
            masks = masks.to(device)

            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, masks)
                preds = outputs >= threshold  # noqa

            running_loss += loss.item()

            preds = preds.type(torch.uint8)  # noqa
            masks = masks.type(torch.uint8)
            meter.update(masks, preds)

            total_count += images.shape[0]

    epoch_loss = running_loss / total_count
    metrics = meter.get_metrics()

    return epoch_loss, metrics


def train_epoch(model, criterion, optimizer, loader, metrics, device, scheduler=None, threshold=0.5):
    model.train()

    total_count = 0
    running_loss = torch.tensor(0.0, dtype=torch.double)
    meter = AverageMetricsMeter(metrics, device)

    with tqdm(loader, desc='Training...') as tqdm_loader: # noqa
        for i, batch in enumerate(tqdm_loader):
            images, masks = batch[:2]

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images) # noqa
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            batch_loss = loss.item()
            running_loss += batch_loss

            preds = (outputs >= threshold).type(torch.uint8)  # noqa
            masks = masks.type(torch.uint8)
            meter.update(masks, preds)

            total_count += images.shape[0]

    epoch_loss = running_loss / total_count
    metrics = meter.get_metrics()

    return epoch_loss, metrics


def print_metrics(metrics):
    print(
        '{phase} metrics:\n'
        'Loss: {loss}\n'
        'IoU: {iou_score}\n\n'.format(**metrics)
    )


def write_metrics(writer, metrics, epoch):
    phase = metrics['phase']
    writer.add_scalars("epoch/{}".format('loss'), {phase: metrics['loss']}, epoch)
    writer.add_scalars("epoch/{}".format('iou'), {phase: metrics['iou_score']}, epoch)


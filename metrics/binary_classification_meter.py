import torch


class BinaryClassificationMeter:
    def __init__(self, device):
        self.conf_mat = torch.zeros(2, 2).to(device)
        self.running_corrects = torch.tensor(0.0, dtype=torch.double)
        self.total_count = torch.tensor(0)

    def update(self, true, pred):
        tp = ((true == 1) & (pred == 1)).sum()
        fn = ((true == 1) & (pred == 0)).sum()
        tn = ((true == 0) & (pred == 0)).sum()
        fp = ((true == 0) & (pred == 1)).sum()

        self.conf_mat += torch.stack([tp, fn, tn, fp]).reshape(2, 2)
        self.running_corrects += (tp + fp)
        self.total_count += true.size(0)  # batch size

    def get_metrics(self):
        tp, fn, tn, fp = self.conf_mat.flatten()

        accuracy = self.running_corrects / self.total_count
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        metrics = {
            'conf_mat': self.conf_mat,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return metrics

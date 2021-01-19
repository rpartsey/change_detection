import segmentation_models_pytorch as smp
from . import losses


def make_loss(config):
    loss_type = getattr(smp.losses, config.type, None) or getattr(losses, config.type)
    loss = loss_type(
        **(config.params if config.params else {})
    )

    return loss

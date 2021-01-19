import segmentation_models_pytorch as smp
from . import models


def make_model(config):
    model_init = getattr(smp, config.type, None) or getattr(models, config.type, None)
    model = model_init(**config.params)

    return model

# todo
from typing import Optional

import timm

from log_infra import def_logger

logger = def_logger.getChild(__name__)


def _build_timm(name: str,
                pretrained: bool,
                weights_path: Optional[str] = "",
                num_classes: Optional[int] = None,
                force_reset_head: bool = False):
    """
    :param name: Name of the model
    :param pretrained: Whether to load pretrained weights
    :param weights_path: Path to custom weights
    :param num_classes: Number of classes for the new head
    :param force_reset_head: Resets classificaiton head even if the number of classes is the same as the pretrained head
    """
    if weights_path and not pretrained:
        logger.warning("Not loading weights from path because pretrained is set to False")

    model = timm.create_model(name, pretrained=pretrained)
    num_classes = num_classes or model.num_classes
    if num_classes != model.num_classes or force_reset_head:
        logger.info(f"Resetting head to {num_classes} classes")
        model.reset_classifier(num_classes=num_classes)
    return model

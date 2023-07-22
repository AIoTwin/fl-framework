"""
    Note: I'm not happy with how preparing datasets and
"""

import torchvision

from log_infra import def_logger

# register custom datasets and transforms
DATASET_DICT = dict()
DATASET_DICT.update(torchvision.datasets.__dict__)

logger = def_logger.getChild(__name__)


def register_dataset(cls_or_func):
    DATASET_DICT[cls_or_func.__name__] = cls_or_func
    return cls_or_func

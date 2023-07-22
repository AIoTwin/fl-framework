import os
from typing import List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import DataParallel, ModuleList, Sequential
from torch.nn.parallel import DistributedDataParallel

from log_infra import def_logger
from misc.config_models import ModelZooConfig
from models.zoo.torch_image_models import _build_timm
from models.zoo.vision import OFFICIAL_MODEL_DICT

MODEL_CLASS_DICT = dict()
MODEL_FUNC_DICT = dict()

logger = def_logger.getChild(__name__)


def register_model_class(cls):
    MODEL_CLASS_DICT[cls.__name__] = cls
    return cls


def register_model_func(func):
    MODEL_FUNC_DICT[func.__name__] = func
    return func


def get_model(model_name,
              repo_or_dir=None,
              frozen_module_paths: Optional[List[str]] = None,
              **kwargs) -> nn.Module:
    model = None
    if model_name in MODEL_CLASS_DICT:
        model = MODEL_CLASS_DICT[model_name](**kwargs)
    elif model_name in MODEL_FUNC_DICT:
        model = MODEL_FUNC_DICT[model_name](**kwargs)
    elif repo_or_dir is not None:
        model = torch.hub.load(repo_or_dir, model_name, **kwargs)
    if model is None:
        raise ValueError('model_name `{}` is not expected'.format(model_name))

    if frozen_module_paths:
        freeze_module_paths(model, frozen_module_paths)

    return model


@register_model_func
def get_torchvision_model(zoo_config):
    model_name = zoo_config['name']
    return OFFICIAL_MODEL_DICT[model_name](**zoo_config['params'])


@register_model_func
def get_torch_image_model(name, params):
    return _build_timm(name, **params)


def init_trainable_weights(m: nn.Module):
    if m.weight.requires_grad:
        return
    if isinstance(m, nn.Linear):
        # Apply custom initialization for Linear layers that are trainable
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    else:
        logger.warning(f"Initialization for layer type: {m.__class__.__name__} not implemented")


def get_torch_module(root_module: nn.Module, module_path):
    module_names = module_path.split(".")
    module = root_module
    for module_name in module_names:
        if not hasattr(module, module_name):
            if isinstance(module, (DataParallel, DistributedDataParallel)):
                module = module.module
                if not hasattr(module, module_name):
                    if (
                            isinstance(module, Sequential)
                            and module_name.lstrip("-").isnumeric()
                    ):
                        module = module[int(module_name)]
                    else:
                        logger.info(
                            f"`{module_name}` of `{module_path}` could not be reached in `{type(root_module).__name__}`"
                        )
                else:
                    module = getattr(module, module_name)
            elif (
                    isinstance(module, (Sequential, ModuleList))
                    and module_name.lstrip("-").isnumeric()
            ):
                module = module[int(module_name)]
            else:
                logger.info(
                    f"`{module_name}` of `{module_path}` could not be reached in `{type(root_module).__name__}`"
                )
                return None
        else:
            module = getattr(module, module_name)
    return module


def freeze_module_paths(model: nn.Module, frozen_module_paths: List[str]) -> None:
    isinstance_str = 'instance('
    for frozen_module_path in frozen_module_paths:
        if frozen_module_path.startswith(isinstance_str) and frozen_module_path.endswith(')'):
            target_cls = nn.__dict__[frozen_module_path[len(isinstance_str):-1]]
            for m in model.modules():
                if isinstance(m, target_cls):
                    freeze_module_params(m)
        else:
            module = get_torch_module(model, frozen_module_path)
            freeze_module_params(module)


def freeze_module_params(modules: Union[List[nn.Module], nn.Module]) -> None:
    modules = modules if isinstance(list, modules) else [modules]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False


def load_model_from_zoo(
        zoo_config: ModelZooConfig, device: str, skip_ckpt: bool = True
) -> nn.Module:
    model = get_model(
        model_name=zoo_config.zoo_or_custom_model_name,
        repo_or_dir=zoo_config.repo_or_dir,
        **zoo_config.model_args,
    )
    if not skip_ckpt:
        ckpt_file_path = os.path.expanduser(zoo_config.ckpt)
        raise NotImplemented("TODO Restore from ckpt")
        # load_ckpt(ckpt_file_path, model=model, strict=True)
    else:
        logger.info("Skipping loading from checkpoint...")
    return model.to(device)


@register_model_class
class NaiveNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(NaiveNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

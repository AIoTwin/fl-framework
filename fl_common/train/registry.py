from typing import Any, Callable, Dict, Optional, Tuple

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import misc.util as misc_util
from models.registry import get_torch_module

OPTIM_DICT = misc_util.get_classes_as_dict('torch.optim')
SCHEDULER_DICT = misc_util.get_classes_as_dict('torch.optim.lr_scheduler')
LOSS_DICT = misc_util.get_classes_as_dict('torch.nn.modules.loss')


def register_optimizer(clbl: Callable):
    OPTIM_DICT[clbl.__name__] = clbl
    return clbl


def register_scheduler(cls_or_func):
    SCHEDULER_DICT[cls_or_func.__name__] = cls_or_func
    return cls_or_func


def _get_loss(loss_type, param_dict=None, **kwargs):
    if param_dict is None:
        param_dict = dict()
    lower_loss_type = loss_type.lower()
    if lower_loss_type in LOSS_DICT:
        return LOSS_DICT[lower_loss_type](**param_dict, **kwargs)
    raise ValueError('loss_type `{}` is not expected'.format(loss_type))


def _get_optimizer(module,
                   optim_type,
                   param_dict=None,
                   filters_params=True,
                   **kwargs) -> Optimizer:
    if param_dict is None:
        param_dict = dict()

    is_module = isinstance(module, nn.Module)
    lower_optim_type = optim_type.lower()
    if lower_optim_type in OPTIM_DICT:
        optim_cls_or_func = OPTIM_DICT[lower_optim_type]
        if is_module and filters_params:
            params = module.parameters() if is_module else module
            updatable_params = [p for p in params if p.requires_grad]
            return optim_cls_or_func(updatable_params, **param_dict, **kwargs)
        return optim_cls_or_func(module, **param_dict, **kwargs)
    raise ValueError('optim_type `{}` is not expected'.format(optim_type))


def _get_scheduler(optimizer, scheduler_type, param_dict=None, **kwargs) -> LRScheduler:
    if param_dict is None:
        param_dict = dict()

    lower_scheduler_type = scheduler_type.lower()
    if lower_scheduler_type in SCHEDULER_DICT:
        return SCHEDULER_DICT[lower_scheduler_type](optimizer, **param_dict, **kwargs)
    raise ValueError(f"scheduler_type `{scheduler_type}` is not expected")


def get_optimizer_and_scheduler(model: nn.Module,
                                optimizer_config: Dict[str, Any],
                                scheduler_config: Optional[Dict[str, Any]] = None) \
        -> Tuple[Optimizer, Optional[LRScheduler]]:
    optim_params_config = optimizer_config["params"]

    module_wise_params_configs = optimizer_config.get('module_wise_params', list())
    if len(module_wise_params_configs) > 0:
        trainable_module_list = list()
        for module_wise_params_config in module_wise_params_configs:
            module_wise_params_dict = dict()
            if isinstance(module_wise_params_config.get("params", None), dict):
                module_wise_params_dict.update(module_wise_params_config["params"])
            module = get_torch_module(model, module_wise_params_config["module"])
            module_wise_params_dict['params'] = module.parameters()
            trainable_module_list.append(module_wise_params_dict)
    else:
        trainable_module_list = nn.ModuleList([model])

    filters_params = optimizer_config.get("filters_params", True)
    optimizer = _get_optimizer(trainable_module_list,
                               optimizer_config["type"],
                               optim_params_config,
                               filters_params)
    optimizer.zero_grad()
    scheduler = None
    if scheduler_config:
        scheduler = _get_scheduler(optimizer, scheduler_config['type'], scheduler_config['params'])
    return optimizer, scheduler


def get_criterion(criterion_config: Dict[str, Any]) -> Callable:
    # todo: Support for complex criterion and custom loss
    return _get_loss(criterion_config["type"], criterion_config["params"])

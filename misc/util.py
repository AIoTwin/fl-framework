import importlib
import inspect
import os
import shutil
import sys
import uuid
from collections import OrderedDict
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Union

import numpy as np
import torch
from torch import Tensor, nn

from log_infra import def_logger

logger = def_logger.getChild(__name__)

PathLike = Union[str, os.PathLike]


def check_if_exists(file_path: PathLike) -> bool:
    return file_path is not None and os.path.exists(file_path)


def get_pymodule(module_path: PathLike) -> ModuleType:
    """
    Return a module reference
    """
    module_ = importlib.import_module(module_path)
    return module_


def get_classes(package_name: str, require_names: bool = False) -> List[Callable]:
    members = inspect.getmembers(sys.modules[package_name], inspect.isclass)
    if require_names:
        return members
    return [obj for _, obj in members]


def get_classes_as_dict(
        package_name: str, to_lower: bool = True
) -> Dict[str, Callable]:
    members = get_classes(package_name, require_names=True)
    class_dict = dict()
    for name, cls in members:
        class_dict[name.lower() if to_lower else name] = cls
    return class_dict


def short_uid() -> str:
    return str(uuid.uuid4())[0:8]


def check_if_torch_module_exits(module: nn.Module, module_path: PathLike) -> bool:
    module_names = module_path.split(".")
    child_module_name = module_names[0]
    if len(module_names) == 1:
        return hasattr(module, child_module_name)

    if not hasattr(module, child_module_name):
        return False
    return check_if_torch_module_exits(
        getattr(module, child_module_name), ".".join(module_names[1:])
    )


def extract_entropy_bottleneck_module(model) -> nn.Module:
    model_wo_ddp = model.module if module_util.check_if_wrapped(model) else model
    entropy_bottleneck_module = None
    if check_if_torch_module_exits(
            model_wo_ddp, "compression_module.entropy_bottleneck"
    ):
        entropy_bottleneck_module = module_util.get_module(
            model_wo_ddp, "compression_module"
        )
    elif check_if_torch_module_exits(
            model_wo_ddp, "compression_model.entropy_bottleneck"
    ):
        entropy_bottleneck_module = module_util.get_module(
            model_wo_ddp, "compression_model"
        )
    return entropy_bottleneck_module


def chmod_r(path: PathLike, mode: int):
    """Recursive chmod"""
    if not os.path.exists(path):
        return
    os.chmod(path, mode)
    for root, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            os.chmod(os.path.join(root, dirname), mode)
        for filename in filenames:
            os.chmod(os.path.join(root, filename), mode)


def rm_rf(path: PathLike):
    """
    Recursively removes a file or directory
    """
    if not path or not os.path.exists(path):
        return
    try:
        chmod_r(path, 0o777)
    except PermissionError:
        pass
    exists_but_non_dir = os.path.exists(path) and not os.path.isdir(path)
    if os.path.isfile(path) or exists_but_non_dir:
        os.remove(path)
    else:
        shutil.rmtree(path)


def recursive_vars(obj):
    """
        Recursive vars

    """
    if isinstance(obj, dict):
        return {key: recursive_vars(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        return recursive_vars(vars(obj))
    elif isinstance(obj, list):
        return [recursive_vars(item) for item in obj]
    else:
        return obj


def mkdir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


class Tokenizer(nn.Module):
    """
    Patch embed without Projection (From Image Tensor to Token Tensor)
    """

    def __init__(self):
        super(Tokenizer, self).__init__()

    def forward(self, x) -> Tensor:
        x = x.flatten(2).transpose(1, 2)  # B h*w C
        return x


class Detokenizer(nn.Module):
    """
    Inverse operation of Tokenizer (From Token Tensor to Image Tensor)
    """

    def __init__(self, spatial_dims):
        super(Detokenizer, self).__init__()
        self.spatial_dims = spatial_dims

    def forward(self, x) -> Tensor:
        B, _, C = x.shape
        H, W = self.spatial_dims
        return x.transpose(1, 2).view(B, -1, H, W)


def overwrite_config(org_config: Dict[str, Any], sub_config: Dict[str, Any]):
    def _isfloat(num):
        try:
            float(num)
            return True
        except ValueError:
            return False

    for sub_key, sub_value in sub_config.items():
        if sub_key in org_config:
            if isinstance(sub_value, dict):
                overwrite_config(org_config[sub_key], sub_value)
            else:
                org_config[sub_key] = (
                    float(sub_value) if _isfloat(sub_value) else sub_value
                )
        else:
            org_config[sub_key] = sub_value


def ndarray_to_weight_dict(
        keys: Iterable[str], params: List[np.ndarray]
) -> Dict[str, nn.Parameter]:
    params_dict = zip(keys, params)
    return OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})

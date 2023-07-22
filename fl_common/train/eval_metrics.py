import functools
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from log_infra import MetricLogger, def_logger

logger = def_logger.getChild(__name__)

_EVAL_METRICS_REGISTRY = dict()


@torch.inference_mode()
def compute_accuracy(outputs, targets, topk=(1,), include_ce: Optional[bool] = False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = targets.size(0)
    _, preds = outputs.topk(maxk, 1, True, True)
    preds = preds.t()
    corrects = preds.eq(targets[None])
    result_list = []
    for k in topk:
        correct_k = corrects[:k].flatten().sum(dtype=torch.float32)
        result_list.append(correct_k * (100.0 / batch_size))
    if include_ce:
        result_list.append(F.cross_entropy(outputs, targets))
    return result_list


def register_eval_metric(_func: Callable = None, *, name: Optional[str] = None):
    def decorator_register(cls):
        @functools.wraps(cls)
        def wrapper_register():
            cls_name = name or cls.__name__
            _EVAL_METRICS_REGISTRY[cls_name] = cls

        wrapper_register()
        return cls

    if _func is None:
        return decorator_register
    else:
        return decorator_register(_func)


class EvaluationMetric(ABC):
    """
        Todo: Using class methods to keep track of best_val and init_best_val would not due to tracking multiple clients
            However, keeping an internal dict of best_val and init_best_val with client_id as key would be cool
    """
    @abstractmethod
    def comparator(self, val1: float, val2: float) -> bool:
        raise NotImplementedError

    # Let's throw any understanding of OOP out of the window
    @staticmethod
    @abstractmethod
    def eval_func(model: nn.Module,
                  data_loader: DataLoader,
                  device: str,
                  title: Optional[str],
                  header: str = "Validation:",
                  *args,
                  **kwargs) -> Union[float, Dict[str, float]]:
        raise NotImplementedError

    def reset(self):
        self.val = self.init_best_val

    @property
    def best_val(self) -> float:
        return self.best_val

    @best_val.setter
    def best_val(self, val: float):
        self._best_val = val

    @property
    @abstractmethod
    def init_best_val(self) -> float:
        raise NotImplementedError

    @init_best_val.setter
    def init_best_val(self, val: float):
        self._init_best_val = val


@register_eval_metric(name="Accuracy")
class AccuracyEvaluation(EvaluationMetric):
    def __init__(self, eval_args: Optional[Dict[str, Any]] = None):
        if eval_args:
            self.eval_func = partial(self.eval_func, **eval_args)
        self.init_best_val = 0

    def comparator(self, val1: float, val2: float) -> bool:
        return val1 > val2

    @property
    def init_best_val(self) -> float:
        return 0

    @init_best_val.setter
    def init_best_val(self, val: float):
        self._init_best_val = val

    @staticmethod
    @torch.inference_mode()
    def eval_func(model: nn.Module,
                  data_loader: DataLoader,
                  device: str,
                  log_freq=1000,
                  title: Optional[str] = None,
                  header: str = "Validation:",
                  include_ce: Optional[bool] = True,
                  **kwargs) -> Union[float, Dict[str, float]]:
        model.eval()
        metric_logger = MetricLogger(delimiter='  ')
        for image, target in metric_logger.log_every(data_loader, log_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            acc1, acc5, ce = compute_accuracy(output, target,
                                              topk=(1, 5),
                                              include_ce=True)
            batch_size = image.shape[0]
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["ce"].update(ce.item(), n=batch_size)

        # gather the stats from all processes
        top1_accuracy = metric_logger.acc1.global_avg
        top5_accuracy = metric_logger.acc5.global_avg
        cross_entropy = metric_logger.ce.global_avg
        logger.info(" * Acc@1 {:.4f}\tAcc@5 {:.4f}\tCross Entropy: {:.4f} \n".format(top1_accuracy,
                                                                                     top5_accuracy,
                                                                                     cross_entropy))
        if include_ce:
            return {"acc@1": top1_accuracy, "Cross Entropy": cross_entropy}
        return top1_accuracy


# todo: Support for multiple metrics in Mapping
def get_eval_metric(metric_name: str, **kwargs) -> EvaluationMetric:
    if metric_name not in _EVAL_METRICS_REGISTRY:
        raise ValueError(f"Evaluation metric `{metric_name}` not registered")
    return _EVAL_METRICS_REGISTRY[metric_name]()


def get_eval_metrics(metric_names: Mapping[str, EvaluationMetric]):
    eval_metrics = dict()
    for metric in metric_names:
        eval_metrics[metric] = get_eval_metric(metric)
    return eval_metrics

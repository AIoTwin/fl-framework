import datetime
import time
from collections import defaultdict, deque
from typing import Any, Iterable, Mapping, Optional

import torch
import wandb
from torch import Tensor

from log_infra.log_utils import def_logger
from log_infra.log_visualizers import get_log_visualizers
from log_infra.wandblogger import WandbLogger
from misc.config_models import WandBConfig

py_logger = def_logger.getChild(__name__)


class SmoothedValue:
    """
    Track a series of values and provide access to smoothed values over a window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        pass

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            assert isinstance(
                v, (float, int)
            ), f"`{k}` ({v}) should be either float or int"
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, log_freq, header=None):
        i = 0
        if not header:
            header = ""

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )

        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % log_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    py_logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    py_logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )

            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        py_logger.info("{} Total time: {}".format(header, total_time_str))


class WandBMetricLogger:
    def __init__(
            self,
            wandb_remote_config: Mapping[str, Any],
            disable_wandb: bool,
            delimiter: str ="\t",
            scalar_freq: int = 1000,
    ):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.wandblogger = WandbLogger(wandb_init_config=wandb_remote_config,
                                       disabled=disable_wandb)
        self.step = 0
        self.scalar_freq = scalar_freq
        # self.viz_freq = wandb_remote_config["viz_freq"]
        # self.visualizers = get_log_visualizers(wandb_remote_config["visualizers"])

    def update(
            self, scalars: Mapping[str, Tensor], io_dict: Optional[Mapping[str, Any]] = None
    ):
        """
        Update internal state of metrics. Log to wandb according to configured frequency
        """
        for k, v in scalars.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().item()

            assert isinstance(
                v, (float, int)
            ), f"`{k}` ({v}) should be either float or int"
            self.meters[k].update(v)
        # reminder: this is crap, fix
        if (self.step + 1) % self.scalar_freq == 0:
            self.wandblogger.log(log_dict=scalars, prefix="train")

    def clear(self):
        """
        Reset tracked metrics
        """
        self.step = 0
        del self.meters
        self.meters = defaultdict(SmoothedValue)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def finish(self):
        py_logger.info("Finishing wandb run...")
        wandb.finish(quiet=False)
        time.sleep(60)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, log_freq: int = None, header: str = None):
        i = 0
        if not header:
            header = ""

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )

        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            self.step += 1
            if i % log_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    py_logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    py_logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )

            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        py_logger.info("{} Total time: {}".format(header, total_time_str))


def build_wandb_metric_logger(
        wandb_config: WandBConfig,
        experiment_config_to_log: Mapping[str, Any],
        include_top_levels: Optional[Iterable[str]] = None,
        defer_init: bool = True,
        run_postfix: Optional[str] = None,
) -> WandBMetricLogger:
    """
    If WandBConfig.enabled is set, it initializes a WandB run.
        Then, calls to WandBMetricLogger will relay logs to wandb.
    If wandb is not enabled, the calls are simple no-ops
    """

    if include_top_levels:
        reduced_experiment_config_to_log = {
            top_level: experiment_config_to_log[top_level]
            for top_level in include_top_levels
        }
    else:
        reduced_experiment_config_to_log = experiment_config_to_log
    wandb_remote_config = {**wandb_config.__dict__, "config": reduced_experiment_config_to_log}
    # todo: read Wandb.init config params as dict delete by set intersection
    del wandb_remote_config["enabled"]
    del wandb_remote_config["scalar_freq"]
    if run_postfix:
        wandb_remote_config["name"] = f"´{wandb_remote_config['run_name']}-{run_postfix}"
    del wandb_remote_config["run_name"]
    metric_logger = WandBMetricLogger(wandb_remote_config=wandb_remote_config,
                                      disable_wandb=not wandb_config.enabled,
                                      )
    if not defer_init:
        metric_logger.wandblogger.init()
        metric_logger.wandblogger.define_metric(
            prefix_path="validation/*", metric_name="epoch"
        )
    return metric_logger

from typing import Any, List, Mapping, Optional

import torch
import wandb
from PIL import Image

from misc.config_models import WandBConfig


class WandbLogger:
    def __init__(self,
                 wandb_init_config: Mapping[str, Any],
                 disabled: bool) -> None:

        self.disabled = disabled
        self.init_config = wandb_init_config
        self.entries = dict()
        self._enabled = False

    @property
    def enabled(self):
        return not self.disabled and self._enabled

    @enabled.setter
    def enabled(self, enable: bool):
        self._enabled = enable

    def define_metric(self, prefix_path: str, metric_name: str):
        """
           Add custom metric for the x-axis on the dashboard plots to the prefix path to override the default "step"
        """
        if self.enabled:
            wandb.define_metric(prefix_path, step_metric=metric_name)

    def init(self):
        if not self.disabled:
            wandb.init(**self.init_config)
            self.enabled = True
        # wandb.run.name = wandb.run.id

    def watch_model(self, model: torch.nn.Module):
        if self.enabled:
            wandb.watch(model, log="all", log_freq=100, log_graph=False)

    def log(self, log_dict: dict, prefix: Optional[str] = None):
        if self.enabled:
            wandb.log({f"{prefix}/{k}": v for k, v in log_dict.items()} if prefix else log_dict)

    def set_config_value(self, key: str, value):

        if self.enabled:
            wandb.config[key] = value

    def log_img(self, key: str, img: Image.Image, caption: str = ''):

        if self.enabled:
            wandb.log({key: wandb.Image(img, caption=caption)})

    def log_table(
            self, key: str, columns: List[str], data):
        if self.enabled:
            table = wandb.Table(columns=columns)
            entries = self.entries.get(key, list())
            entries.append(data)
            # this sucks so f***ing bad, but atm you cannot append to wandb tables
            for entry in entries:
                for data in entry:
                    table.add_data(*data)
            self.entries[key] = entries
            wandb.log({key: table}, )

    def finish(self):
        if self.enabled:
            wandb.finish()

import time
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset

from data_retrieval import DatasetContainer, retrieval
from data_retrieval.samplers.indexer import build_indices
from fl_common.train.eval_metrics import get_eval_metrics
from fl_common.train.registry import get_criterion, get_optimizer_and_scheduler
from log_infra import def_logger
from log_infra.metric_logging import SmoothedValue, WandBMetricLogger
from misc.config_models import TrainerConfig

logger = def_logger.getChild(__name__)


# todo: make registrable and retrievable

class ClientTrainer:
    def __init__(
            self,
            device: str,
            trainer_configuration: TrainerConfig,
            metric_logger: WandBMetricLogger,
            datasets_dict: Dict[str, Dataset],
            client_id: int,
    ):
        self.device = device
        self.train_configuration = trainer_configuration.train_config
        self.validation_configuration = trainer_configuration.validation_config
        self.test_configuration = trainer_configuration.test_config
        self.metric_logger = metric_logger
        train_set = datasets_dict[self.train_configuration.train_dataset_id]
        test_set = datasets_dict[self.validation_configuration.eval_dataset_id]
        sampler_indices = build_indices(strategy="flat_fair",
                                        data_source=train_set,
                                        rank=client_id,
                                        world_size=2,
                                        n_classes=10)

        # todo: iterate through each dataset in the dataset dict, and use the dataset_id as the key
        #  check whether there are indices for each set
        self.loaders = {
            "train": retrieval.build_data_loader(
                dataset=train_set,
                sampler_indices=sampler_indices,
                data_loader_config=self.train_configuration.train_loader_params,
            ),
            "validation": retrieval.build_data_loader(
                dataset=test_set,
                data_loader_config=self.validation_configuration.eval_loader_params,
            )
        }
        self.set_sizes = {
            "train": len(train_set),
            "test": len(test_set),
        }
        self.ckpt_path = trainer_configuration.ckpt_path
        self.epoch = 0

    @staticmethod
    def train_one_epoch(
            model: nn.Module,
            metric_logger: WandBMetricLogger,
            train_loader: DataLoader,
            optimizer: Optimizer,
            scheduler: Optional[LRScheduler],
            criterion: nn.Module,
            device: str,
            log_freq: int,
            epoch: int,
            precision: Optional[str] = None,
    ):
        # todo: scheduling, mixed precision training, gradient accumulation and pruning, etc.
        model.to(device)
        model.train()
        metric_logger.add_meter(
            name=f"lr", meter=SmoothedValue(window_size=1, fmt="{value}")
        )
        metric_logger.add_meter(
            name="img/s", meter=SmoothedValue(window_size=10, fmt="{value:.2f}")
        )
        for samples, targets in metric_logger.log_every(
                train_loader, log_freq=log_freq, header=f"Epoch: [{epoch}]"
        ):
            samples, targets = samples.to(device), targets.to(device)
            start_time = time.time()
            batch_size = samples.shape[0]
            optimizer.zero_grad()
            predictions = model(samples)
            loss = criterion(predictions, targets)
            scalars = dict()
            if isinstance(loss, dict):  # loss term may have multiple subterms
                total_loss = 0
                for loss_name, loss_val in loss.items():
                    total_loss += loss_val
                    scalars[loss_name] = loss_val.detach().item()
                scalars["total_loss"] = total_loss
                loss = total_loss
            else:
                scalars["loss"] = loss.item()

            scalars["lr"] = optimizer.param_groups[0]["lr"]
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            metric_logger.update(scalars=scalars)
            metric_logger.meters["img/s"].update(
                batch_size / (time.time() - start_time)
            )
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError("Detected faulty loss = {}".format(loss))

    def train(self, model):
        optimizer, scheduler = get_optimizer_and_scheduler(
            model=model,
            optimizer_config=self.train_configuration.optimizer,
            scheduler_config=self.train_configuration.scheduler,
        )
        max_grad_norm = self.train_configuration.max_grad_norm
        grad_accum_step = self.train_configuration.grad_accum_steps
        criterion = get_criterion(self.train_configuration.criterion)
        iter_wise_scheduler = False  # todo: check if epoch or iterwise scheduler
        for epoch in range(self.train_configuration.epochs):
            self.epoch = epoch
            # pre epoch processing
            ClientTrainer.train_one_epoch(
                model=model,
                metric_logger=self.metric_logger,
                train_loader=self.loaders.get("train"),
                optimizer=optimizer,
                scheduler=scheduler if iter_wise_scheduler else None,
                criterion=criterion,
                device=self.device,
                log_freq=self.train_configuration.log_freq,
                epoch=epoch,
            )
            # post epoch processing

    def validate(self, model: nn.Module, test_mode: bool = False) -> Dict[str, float]:

        eval_config = (
            self.test_configuration
            if self.test_configuration and test_mode
            else self.validation_configuration
        )
        eval_metrics = eval_config.eval_metrics
        if isinstance(eval_metrics, str):
            eval_metrics = [eval_metrics]
        eval_metrics = get_eval_metrics(eval_metrics)
        result_dict = dict({"epoch": self.epoch})
        prefix = "testing" if test_mode else "validation"
        for name, eval_metric in eval_metrics.items():
            res = eval_metric.eval_func(
                model=model,
                data_loader=(self.loaders["testing"]
                             if test_mode and "testing" in self.loaders
                             else self.loaders["validation"]),
                device=self.device,
                log_freq=eval_config.log_freq,
                title=f"{prefix.capitalize()} {name}:",
                header="Test" if test_mode else "Validation",
            )
            if isinstance(res, dict):
                result_dict.update(res)
            else:
                result_dict[name] = res
        self.metric_logger.wandblogger.log(
            {f"{prefix}/{k}" if k != "epoch" else k: v for k, v in result_dict.items()}
        )
        return result_dict

    def test(self, model: nn.Module) -> Dict[str, float]:
        result_dict = self.validate(model=model, test_mode=True)
        # additional post training processing and evaluation
        return result_dict

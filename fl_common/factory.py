"""

"""
import flwr as fl

from typing import Dict, Optional

from torch import nn
from torch.utils.data import Dataset

from fl_common.clients import FlowerBaseClient, get_flower_client
from fl_common.servers.registry import get_flower_server
from fl_common.train.trainer import ClientTrainer
from log_infra import WandBMetricLogger
from misc.config_models import ClientConfig, ServerConfig, StrategyConfig


def create_server(
        server_config: ServerConfig,
        strategy_config: StrategyConfig,
        constructed_model: nn.Module,
        wandb_metric_logger: WandBMetricLogger,
        test_set: Dataset,
        device: str,
        parent_address: Optional[str] = None,
        *args,
        **kwargs
):
    # todo: support for registering and retrieving strategy
    #  strategy = get_fl_strategy(name=strategy_config.strategy_type, ...)

    strategy = fl.server.strategy.FedAvg(**strategy_config.strategy_params)

    server_cls = get_flower_server(name=server_config.server_type)
    kwargs.update(server_config.server_kwargs)
    server = server_cls(
        model=constructed_model,
        test_config=server_config.central_test_config,
        device=device,
        strategy=strategy,
        wandb_metric_logger=wandb_metric_logger,
        test_set=test_set,
        server_address=server_config.server_address,
        rounds=server_config.rounds,
        parent_address=parent_address,
    )
    return server


def create_client(
        client_config: ClientConfig,
        client_id: int,
        client_trainer: ClientTrainer,
        constructed_model: nn.Module
) -> FlowerBaseClient:
    client_cls = get_flower_client(name=client_config.client_type)
    client = client_cls(
        model=constructed_model,
        client_trainer=client_trainer,
        client_id=client_id,
        server_address=client_config.server_address,
        **client_config.client_params)
    return client

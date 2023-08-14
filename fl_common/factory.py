import threading
from functools import partial

import flwr as fl

from typing import Any, Dict, Optional

from torch import nn
from torch.utils.data import Dataset

from data_retrieval import get_all_datasets
from fl_common.aggregators.registry import Aggregator, AggregatorParentConnection
from fl_common.clients import get_flower_client
from fl_common.clients.registry import TorchBaseClient
from fl_common.servers.registry import get_flower_server
from fl_common.strategy.registry import get_strategy
from fl_common.train.trainer import ClientTrainer
from log_infra import WandBMetricLogger, build_wandb_metric_logger
from misc.config_models import (
    AggregatorConfig,
    ClientConfig,
    DatasetsConfig,
    ModelZooConfig,
    StrategyConfig,
    WandBConfig,
)
from misc.util import IterableSimpleNamespace, recursive_vars
from models.registry import load_model_from_zoo


def _create_model(device: str, zoo_config: Dict[str, Any]) -> nn.Module:
    return load_model_from_zoo(device=device, zoo_config=zoo_config)


def _create_wandb_metric_logger(
    wandb_config: WandBConfig,
    run_postfix: Optional[str] = None,
    config_to_log: Optional[Dict[str, Any]] = None,
    defer_init: bool = False,
) -> WandBMetricLogger:
    return build_wandb_metric_logger(
        wandb_config=wandb_config,
        experiment_config_to_log=recursive_vars(config_to_log),
        run_postfix=run_postfix,
        defer_init=defer_init,
    )


# todo: IPC to have one central place where we load datasets
def _create_datasets(dataset_config: Dict[str, Any]):
    return get_all_datasets(dataset_config)


def create_aggregator(
    aggregator_config: AggregatorConfig,
    model_config: ModelZooConfig,
    wandb_config: WandBConfig,
    datasets_config: DatasetsConfig,
) -> Aggregator:
    wandb_metric_logger = _create_wandb_metric_logger(
        wandb_config=wandb_config,
        run_postfix=f"-server_at={aggregator_config.server_address}",
        config_to_log=aggregator_config,
        defer_init=False,
    )

    test_set = _create_datasets(datasets_config.params)[
        aggregator_config.central_test_config.central_dataset_id
    ]

    strategy_partial_params = {
        **aggregator_config.strategy_config.strategy_params,
        "num_children": aggregator_config.num_children,
    }

    model = _create_model(aggregator_config.device, model_config)
    if aggregator_config.parent_address:
        # create shared state
        server_lock = threading.Lock()
        client_lock = threading.Lock()
        server_lock.acquire()
        client_lock.acquire()
        shared_state = IterableSimpleNamespace(
            **{
                "local_samples_processed": 0,
                "current_local_round": 0,
                "no_local_rounds": aggregator_config.num_children,
                "server_lock": server_lock,
                "client_lock": client_lock,
                "model": model,
            }
        )

        # create strategy
        # todo move to factory
        strategy = get_strategy(
            name=aggregator_config.strategy_config.strategy_type,
            partial_params=strategy_partial_params,
            wrap_with_locking=True,
        )

        strategy = partial(strategy, shared_state=shared_state)

        # create parent connection
        parent_connection = AggregatorParentConnection(
            device=aggregator_config.device,
            test_set=test_set,
            eval_config=aggregator_config.central_test_config,
            parent_address=aggregator_config.parent_address,
            shared_state=shared_state,
            metric_logger=wandb_metric_logger,
        )
    else:
        parent_connection = None
        strategy = get_strategy(
            name=aggregator_config.strategy_config.strategy_type,
            partial_params=strategy_partial_params,
        )
        # todo: remove this by combining Eval stuff in Aggregator
        aggregator_config.server_kwargs.update(
            **{
                "test_set": test_set,
                "wandb_metric_logger": wandb_metric_logger,
                "test_config": aggregator_config.central_test_config,
                "model": model,
                "device": aggregator_config.device,
            }
        )

    aggregator_server = get_flower_server(name=aggregator_config.server_type)(
        **{
            **aggregator_config.server_kwargs,
            "strategy": strategy,
            "server_address": aggregator_config.server_address,
            "rounds": aggregator_config.rounds,
        }
    )
    return Aggregator(server=aggregator_server, parent_connection=parent_connection)


def create_client(
    client_config: ClientConfig,
    model_config: ModelZooConfig,
    datasets_config: DatasetsConfig,
    wandb_config: WandBConfig,
) -> TorchBaseClient:
    client_id = int(client_config.client_id)
    client_wandb_metric_logger = _create_wandb_metric_logger(
        wandb_config=wandb_config,
        config_to_log=recursive_vars(client_config),
        run_postfix=f"-client={client_id}",
        defer_init=False,
    )
    model = _create_model(client_config.device, model_config)
    client_trainer = ClientTrainer(
        trainer_configuration=client_config.trainer_config,
        metric_logger=client_wandb_metric_logger,
        device=client_config.device,
        datasets_dict=get_all_datasets(datasets_config=datasets_config.params),
        client_id=client_id,
    )
    client_cls = get_flower_client(name=client_config.client_type)
    client = client_cls(
        model=model,
        client_trainer=client_trainer,
        client_id=client_id,
        server_address=client_config.server_address,
        **client_config.client_params,
    )
    return client


def create_unreliable_client(
        unreliable_client_config: UnreliableClientConfig,
        client_id: int,
        failure_rate:int,
        client_trainer: ClientTrainer,
        constructed_model: nn.Module
) -> FlowerBaseClient:
    client_cls = get_unreliable_client(name=unreliable_client_config.unreliable_client_type)
    client = client_cls(
        model=constructed_model,
        client_id=client_id,
        client_trainer=client_trainer,
        failure_rate=failure_rate,
        server_address=unreliable_client_config.server_address,
        **unreliable_client_config.client_params)
    return client


# todo: delete
# def create_server(
#         server_config: ServerConfig,
#         strategy_config: StrategyConfig,
#         constructed_model: nn.Module,
#         wandb_metric_logger: WandBMetricLogger,
#         test_set: Dataset,
#         device: str,
#         parent_address: Optional[str] = None,
#         *args,
#         **kwargs
# ):
#     # todo: support for registering and retrieving strategy
#     #  strategy = get_fl_strategy(name=strategy_config.strategy_type, ...)
#
#     strategy = partial(fl.server.strategy.FedAvg, **strategy_config.strategy_params)
#
#     server_cls = get_flower_server(name=server_config.server_type)
#     kwargs.update(server_config.server_kwargs)
#     server = server_cls(
#         model=constructed_model,
#         test_config=server_config.central_test_config,
#         device=device,
#         strategy=strategy,
#         wandb_metric_logger=wandb_metric_logger,
#         test_set=test_set,
#         server_address=server_config.server_address,
#         rounds=server_config.rounds,
#         parent_address=parent_address,
#     )
#     return server

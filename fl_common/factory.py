"""

"""

from torch import nn
from torch.utils.data import Dataset

from fl_common.clients import FlowerBaseClient, get_flower_client
from fl_common.clients.registry import get_unreliable_client
from fl_common.servers.registry import get_flower_server
from fl_common.train.trainer import ClientTrainer
from log_infra import WandBMetricLogger
from misc.config_models import ClientConfig, ServerConfig, UnreliableClientConfig


def create_server(
        server_type: str,
        server_config: ServerConfig,
        constructed_model: nn.Module,
        wandb_metric_logger: WandBMetricLogger,
        test_set: Dataset,
):
    server_cls = get_flower_server(name=server_type)
    server = server_cls(
        model=constructed_model,
        server_config=server_config,
        wandb_metric_logger=wandb_metric_logger,
        test_set=test_set,
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


def create_unreliable_client(
        unreliable_client_config: UnreliableClientConfig,
        client_id: int,
        client_trainer: ClientTrainer,
        constructed_model: nn.Module
) -> FlowerBaseClient:
    client_cls = get_unreliable_client(name=unreliable_client_config.unreliable_client_type)
    client = client_cls(
        model=constructed_model,
        client_id=client_id,
        client_trainer=client_trainer,
        failure_rate=unreliable_client_config.failure_rate,
        server_address=unreliable_client_config.server_address,
        **unreliable_client_config.client_params)
    return client

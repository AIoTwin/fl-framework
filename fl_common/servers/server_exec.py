import sys

import flwr as fl
from spock import SpockBuilder

from data_retrieval.retrieval import get_all_datasets, get_dataset_dict
from fl_common import factory
from fl_common.servers.registry import get_flower_server
from log_infra import build_wandb_metric_logger, def_logger, prepare_local_log_file
from misc.config_models import DatasetsConfig, LoggingConfig, ModelZooConfig, ServerConfig
from misc.util import recursive_vars
from models.registry import load_model_from_zoo

logger = def_logger.getChild(__name__)

"""
  Encapsulate this in object
  
  def start_server(flower_server):
    # server.wandb_metric_logger.wandblogger.init()

    # todo: strategy registry and retrieve it according to strategy_type
    strategy = fl.server.strategy.FedAvg(
        **vars(flower_server.server_config.strategy_config.strategy_params),
        evaluate_fn=flower_server.__class__.evaluate_fn
    )

    logger.info(f"Starting server at address {flower_server.server_address}...")
    fl.server.start_server(
        server_address=flower_server.server_address,
        config=fl.server.ServerConfig(num_rounds=flower_server.server_config.server_params.rounds),
        strategy=strategy
    )
"""

#


if __name__ == '__main__':
    config_space = SpockBuilder(ServerConfig,
                                ModelZooConfig,
                                LoggingConfig,
                                DatasetsConfig,
                                lazy=True,
                                desc="Build and start server").generate()

    logging_config: LoggingConfig = config_space.LoggingConfig
    zoo_config: ModelZooConfig = config_space.ModelZooConfig
    server_config: ServerConfig = config_space.ServerConfig
    datasets_config: DatasetsConfig = config_space.DatasetsConfig
    prepare_local_log_file(log_file_path=logging_config.local_logging_config.log_file_path,
                           mode="a",
                           overwrite=False)
    model = load_model_from_zoo(device=server_config.device, zoo_config=zoo_config)
    server_wandb_metric_logger = build_wandb_metric_logger(
        wandb_config=logging_config.wandb_config,
        experiment_config_to_log=recursive_vars(server_config),
        run_postfix="-server",
        defer_init=False,
    )
    # todo: IPC to have one central place where we load datasets
    test_set = get_all_datasets(datasets_config.params)[server_config.central_test_config.central_dataset_id]
    server = factory.create_server(
        server_type=server_config.server_type,
        server_config=server_config,
        constructed_model=model,
        test_set=test_set,
        wandb_metric_logger=server_wandb_metric_logger,
    )
    server.start()

import sys

import flwr as fl
from spock import SpockBuilder

from data_retrieval.retrieval import get_all_datasets, get_dataset_dict
from fl_common import factory
from fl_common.servers.registry import get_flower_server
from log_infra import build_wandb_metric_logger, def_logger, prepare_local_log_file
from misc.config_models import AggregatorConfig, DatasetsConfig, LoggingConfig, ModelZooConfig, ServerConfig
from misc.util import recursive_vars
from models.registry import load_model_from_zoo

logger = def_logger.getChild(__name__)


def _default_strategy_params(params_dict, no_clients):
    if "min_available_clients" not in params_dict:
        params_dict["min_available_clients"] = no_clients
    if "min_fit_clients" not in params_dict:
        params_dict["min_fit_clients"] = no_clients
    if "min_eval_clients" not in params_dict:
        params_dict["min_evaluate_clients"] = no_clients


if __name__ == '__main__':
    config_space = SpockBuilder(AggregatorConfig,
                                ModelZooConfig,
                                LoggingConfig,
                                DatasetsConfig,
                                lazy=True,
                                desc="Build and start server").generate()

    logging_config: LoggingConfig = config_space.LoggingConfig
    zoo_config: ModelZooConfig = config_space.ModelZooConfig
    datasets_config: DatasetsConfig = config_space.DatasetsConfig
    aggregator_config: AggregatorConfig = config_space.AggregatorConfig
    _default_strategy_params(aggregator_config.strategy_config.strategy_params,
                             aggregator_config.no_children)
    prepare_local_log_file(log_file_path=logging_config.local_logging_config.log_file_path,
                           mode="a",
                           overwrite=False)
    model = load_model_from_zoo(device=aggregator_config.device, zoo_config=zoo_config)
    server_wandb_metric_logger = build_wandb_metric_logger(
        wandb_config=logging_config.wandb_config,
        experiment_config_to_log=recursive_vars(aggregator_config),
        run_postfix=f"-server_address={aggregator_config.server_config.server_address}",
        defer_init=False,
    )
    # todo: IPC to have one central place where we load datasets
    test_set = get_all_datasets(datasets_config.params)[
        aggregator_config.server_config.central_test_config.central_dataset_id]
    server = factory.create_server(
        server_config=aggregator_config.server_config,
        strategy_config=aggregator_config.strategy_config,
        parent_address=aggregator_config.parent_address,
        device=aggregator_config.device,
        constructed_model=model,
        test_set=test_set,
        wandb_metric_logger=server_wandb_metric_logger,
    )
    server.start()

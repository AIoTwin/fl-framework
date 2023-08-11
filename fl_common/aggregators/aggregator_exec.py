import threading

from spock import SpockBuilder

from data_retrieval import get_all_datasets
from fl_common import factory
from fl_common.aggregators.registry import Aggregator, AggregatorParentConnection
from fl_common.servers.registry import get_flower_server
from fl_common.strategy.registry import get_strategy
from log_infra import build_wandb_metric_logger, prepare_local_log_file
from misc.config_models import (
    AggregatorConfig,
    DatasetsConfig,
    LoggingConfig,
    ModelZooConfig,
)
from misc.util import IterableSimpleNamespace, recursive_vars
from models.registry import load_model_from_zoo

if __name__ == "__main__":
    # todo
    config_space = SpockBuilder(
        AggregatorConfig,
        ModelZooConfig,
        LoggingConfig,
        DatasetsConfig,
        lazy=True,
        desc="Build and start server",
    ).generate()

    logging_config: LoggingConfig = config_space.LoggingConfig
    model_config: ModelZooConfig = config_space.ModelZooConfig
    datasets_config: DatasetsConfig = config_space.DatasetsConfig
    aggregator_config: AggregatorConfig = config_space.AggregatorConfig
    prepare_local_log_file(
        log_file_path=logging_config.local_logging_config.log_file_path,
        mode="a",
        overwrite=False,
    )

    aggregator = factory.create_aggregator(
        aggregator_config=aggregator_config,
        wandb_config=logging_config.wandb_config,
        datasets_config=datasets_config,
        model_config=model_config)

    aggregator.start()

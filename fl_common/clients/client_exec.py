import sys

from spock import SpockBuilder

from data_retrieval import get_all_datasets
from fl_common import factory
from fl_common.factory import create_client
from fl_common.train.trainer import ClientTrainer
from log_infra import build_wandb_metric_logger, def_logger, prepare_local_log_file
from misc.config_models import (
    ClientConfig,
    DatasetsConfig,
    LoggingConfig,
    ModelZooConfig,
)
from misc.util import recursive_vars
from models.registry import load_model_from_zoo

logger = def_logger.getChild(__name__)

if __name__ == "__main__":
    config_space = SpockBuilder(
        ClientConfig,
        LoggingConfig,
        ModelZooConfig,
        DatasetsConfig,
        lazy=True,
        desc="Build and start client",
    ).generate()
    logging_config = config_space.LoggingConfig
    model_config = config_space.ModelZooConfig
    client_config = config_space.ClientConfig
    datasets_config = config_space.DatasetsConfig
    prepare_local_log_file(
        log_file_path=logging_config.local_logging_config.log_file_path,
        mode="a",
        overwrite=False,
    )

    client = create_client(
        client_config=client_config,
        model_config=model_config,
        datasets_config=datasets_config,
        wandb_config=logging_config.wandb_config,
    )
    client.start()

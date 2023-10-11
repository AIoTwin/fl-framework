import random

import numpy as np
import torch
from spock import SpockBuilder

from fl_common import factory
from log_infra import prepare_local_log_file
from misc.config_models import (
    AggregatorConfig,
    DatasetsConfig,
    LoggingConfig,
    ModelZooConfig,
)


def set_seed(seed):
    if seed == -1:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    set_seed(model_config.random_seed)
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

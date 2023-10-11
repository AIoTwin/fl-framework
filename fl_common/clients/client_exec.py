import random

import numpy as np
import torch
from spock import SpockBuilder

from fl_common.factory import create_client
from log_infra import def_logger, prepare_local_log_file
from misc.config_models import (
    ClientConfig,
    DatasetsConfig,
    LoggingConfig,
    ModelZooConfig,
)

logger = def_logger.getChild(__name__)


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
    set_seed(model_config.random_seed)
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

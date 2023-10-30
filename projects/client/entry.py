import concurrent.futures
import time
from enum import Enum
from typing import List

from spock import SpockBuilder, spock

from fl_common.process_handler import AGGREGATOR_EXEC, CLIENT_EXEC, run_script
from log_infra import def_logger, prepare_local_log_file
from misc.config_models import ClientConfig, DatasetsConfig, LoggingConfig, ModelZooConfig

logger = def_logger.getChild(__name__)


@spock
class ClientEntryConfig:
    """
    Extend as needed "E.g., clusters: List[ClusterConfig]"
    """
    test_only: bool = False
    client_config: ClientConfig
    model_config: ModelZooConfig
    datasets_config: DatasetsConfig
    logging_config: LoggingConfig


logger = def_logger.getChild(__name__)


def run(root_config: ClientEntryConfig):
    prepare_local_log_file(
        log_file_path=root_config.logging_config.local_logging_config.log_file_path,
        test_only=root_config.test_only,
        overwrite=False,
    )
    client_id = '1'
    logger.info(
        f"Starting FL client"
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=None
    ) as executor:
        executor.submit(
            run_script,
            CLIENT_EXEC,
            [
                "-c",
                "config/example_container/client_config.yaml",
            ])


if __name__ == "__main__":
    description = "FL client entry"
    config = SpockBuilder(
        ClientEntryConfig,
        desc=description,
        lazy=True,
    ).generate()
    # todo: global cuda settings (e.g., cudnn.benchmark) for performance
    run(root_config=config.ClientEntryConfig)

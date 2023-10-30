import concurrent.futures
import time
from enum import Enum
from typing import List

from spock import SpockBuilder, spock

from fl_common.process_handler import AGGREGATOR_EXEC, CLIENT_EXEC, run_script
from log_infra import def_logger, prepare_local_log_file
from misc.config_models import ClientConfig, DatasetsConfig, LoggingConfig

logger = def_logger.getChild(__name__)


class SubsetStrategy(Enum):
    """
    All strategies we want to try out to distribute subsets
    E.g.,
     Each client gets distinct classes but the same number of sample/classes,
     Each client gets a different number of samples/classes
     Clients within a cluster can share classes, etc.

    """

    flat_fair = "flat_fair"


@spock
class GlobalServerEntryConfig:
    """
    Extend as needed "E.g., clusters: List[ClusterConfig]"
    """
    test_only: bool = False
    train_split: int
    datasets_config: DatasetsConfig
    logging_config: LoggingConfig
    subset_strategy: SubsetStrategy


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
                "config/example_client/client_config.yaml",
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

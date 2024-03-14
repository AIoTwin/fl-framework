import concurrent.futures
import time
from enum import Enum
from typing import List

from spock import SpockBuilder, spock

from fl_common.process_handler import AGGREGATOR_EXEC, CLIENT_EXEC, run_script
from log_infra import def_logger, prepare_local_log_file
from misc.config_models import ClientConfig, DatasetsConfig, LoggingConfig

logger = def_logger.getChild(__name__)


@spock
class GlobalServerEntryConfig:
    """
    Extend as needed "E.g., clusters: List[ClusterConfig]"
    """
    test_only: bool = False
    num_clients: int
    logging_config: LoggingConfig
    dataset_config: DatasetsConfig


logger = def_logger.getChild(__name__)


def run(root_config: GlobalServerEntryConfig):
    prepare_local_log_file(
        log_file_path=root_config.logging_config.local_logging_config.log_file_path,
        test_only=root_config.test_only,
        overwrite=False,
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=None
    ) as executor:
        executor.submit(
            run_script,
            AGGREGATOR_EXEC,
            [
                "-c",
                "config/example_global_server/aggregator_config.yaml",
                "--AggregatorConfig.num_children",
                f"{root_config.num_clients}",
                "--StrategyConfig.strategy_params",
                "{'min_available_clients': "
                + f"{root_config.num_clients},"
                + "'min_fit_clients': "
                + f"{root_config.num_clients},"
                + "'min_evaluate_clients': "
                + f"{root_config.num_clients}"
                + "}",
            ],
        )


if __name__ == "__main__":
    description = "FL global server entry"
    config = SpockBuilder(
        GlobalServerEntryConfig,
        desc=description,
        lazy=True,
    ).generate()
    # todo: global cuda settings (e.g., cudnn.benchmark) for performance
    run(root_config=config.GlobalServerEntryConfig)

import concurrent.futures
import time
from enum import Enum
from typing import List

from spock import SpockBuilder, spock

from fl_common.process_handler import AGGREGATOR_EXEC, CLIENT_EXEC, run_script
from log_infra import def_logger, prepare_local_log_file
from misc.config_models import DatasetsConfig, LoggingConfig

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
    flat_skewed = "flat_skewed"


@spock
class PilotExperimentConfig:
    """
    Extend as needed "E.g., clusters: List[ClusterConfig]"
    """

    test_only: bool = False
    num_clients: int
    failures_at_round: List[int] = list()
    logging_config: LoggingConfig
    dataset_config: DatasetsConfig


logger = def_logger.getChild(__name__)


def run(root_config: PilotExperimentConfig):
    prepare_local_log_file(
        log_file_path=root_config.logging_config.local_logging_config.log_file_path,
        test_only=root_config.test_only,
        overwrite=False,
    )
    server_futures = []
    client_futures = []
    num_unreliable_clients = len(root_config.failures_at_round)
    num_reliable_clients = root_config.num_clients - num_unreliable_clients
    logger.info(
        f"Starting run with {root_config.num_clients} where {num_unreliable_clients} are unreliable"
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=None
    ) as executor:
        server_futures.append(
            executor.submit(
                run_script,
                AGGREGATOR_EXEC,
                [
                    "-c",
                    "config/example_pilot/aggregator_config.yaml",
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
        )
        time.sleep(5)
        for client_id in range(root_config.num_clients):
            client_futures.append(
                executor.submit(
                    run_script,
                    CLIENT_EXEC,
                    [
                        "-c",
                        "config/example_pilot/client_config.yaml",
                        "--ClientConfig.client_id",
                        "p"+f"{client_id}",
                    ]
                    + (
                        [
                            "--ClientConfig.client_type",
                            "UnreliableClient",
                            "--ClientConfig.client_params",
                            "{'fail_at_round': "
                            + f"{root_config.failures_at_round[client_id]}"
                            + "}",
                        ]
                        if client_id < num_unreliable_clients
                        else ["--ClientConfig.client_type", "TorchClient"]
                    ),
                )
            )


if __name__ == "__main__":
    description = "Preliminary Experiments to test the effects of client dropouts on predictive strength"
    config = SpockBuilder(
        PilotExperimentConfig,
        desc=description,
        lazy=True,
    ).generate()
    # todo: global cuda settings (e.g., cudnn.benchmark) for performance
    run(root_config=config.PilotExperimentConfig)

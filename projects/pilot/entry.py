import concurrent.futures
import time
from enum import Enum

from spock import SpockBuilder, spock

from data_retrieval import get_all_datasets
from data_retrieval.samplers.indexer import build_indices
from fl_common.threading.process_handler import run_script
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


@spock
class PilotExperimentConfig:
    """
        Extend as needed "E.g., clusters: List[ClusterConfig]"
    """
    test_only: bool = False
    num_clients: int
    logging_config: LoggingConfig
    dataset_config: DatasetsConfig
    subset_strategy: SubsetStrategy


logger = def_logger.getChild(__name__)


def run(root_config: PilotExperimentConfig):
    prepare_local_log_file(log_file_path=root_config.logging_config.local_logging_config.log_file_path,
                           test_only=root_config.test_only,
                           overwrite=False)
    server_futures = []
    client_futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        server_futures.append(executor.submit(run_script,
                                              "fl_common/servers/server_exec.py",
                                              ["-c", "config/example_config/server_config.yaml"]))
        time.sleep(10)  # Todo: can we poll the server to see if it's ready?
        # run_script("fl_common/servers/server_exec.py",["-c", "config/example_config/server_config.yaml"])
        # for client_id in range(run_config.number_clients):
        for client_id in range(2):
            client_futures.append(
                executor.submit(run_script,
                                "fl_common/clients/client_exec.py",
                                ["-c",
                                 "config/example_config/client_config.yaml",
                                 "--ClientConfig.client_id",
                                 f"{client_id}",
                                 ])
            )


if __name__ == "__main__":
    description = "Preliminary Experiments to test the effects of client dropouts on predictive strength"
    config = SpockBuilder(PilotExperimentConfig,
                          desc=description,
                          lazy=True,
                          ).generate()
    # todo: global cuda settings (e.g., cudnn.benchmark) for performance
    run(root_config=config.PilotExperimentConfig)

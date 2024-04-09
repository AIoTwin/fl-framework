import concurrent.futures
import wandb

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
    num_clients: int
    rounds: int
    server_address: str
    wandb_key: str
    logging_config: LoggingConfig


logger = def_logger.getChild(__name__)


def run(root_config: GlobalServerEntryConfig):
    prepare_local_log_file(
        log_file_path=root_config.logging_config.local_logging_config.log_file_path,
        test_only=False,
        overwrite=False,
    )
    wandb.login(key=root_config.wandb_key)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=None
    ) as executor:
        executor.submit(
            run_script,
            AGGREGATOR_EXEC,
            [
                "-c",
                "config/example_global_server/aggregator_config.yaml",
                f"--AggregatorConfig.num_children", f"{root_config.num_clients}",
                f"--AggregatorConfig.rounds", f"{root_config.rounds}",
                f"--AggregatorConfig.server_address", f"{root_config.server_address}",
                f"--AggregatorConfig.server_type", "TorchServerWithCentralizedEval"])


if __name__ == "__main__":
    description = "FL global server entry"
    config = SpockBuilder(
        GlobalServerEntryConfig,
        desc=description,
        lazy=True,
    ).generate()
    # todo: global cuda settings (e.g., cudnn.benchmark) for performance
    run(root_config=config.GlobalServerEntryConfig)

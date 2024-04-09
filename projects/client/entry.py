import concurrent.futures
import wandb

from spock import SpockBuilder, spock

from fl_common.process_handler import AGGREGATOR_EXEC, CLIENT_EXEC, run_script
from log_infra import def_logger, prepare_local_log_file
from misc.config_models import ClientConfig, DatasetsConfig, LoggingConfig

logger = def_logger.getChild(__name__)


@spock
class ClientEntryConfig:
    """
    Extend as needed "E.g., clusters: List[ClusterConfig]"
    """
    client_id: int
    epochs: int
    server_address: str
    wandb_key: str
    logging_config: LoggingConfig


logger = def_logger.getChild(__name__)


def run(root_config: ClientEntryConfig):
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
            CLIENT_EXEC,
            [
                "-c",
                "config/example_client/client_config.yaml",
                f"--ClientConfig.client_id", f"h{root_config.client_id}",
                f"--ClientConfig.server_address", f"{root_config.server_address}",
                f"--TrainConfig.epochs", f"{root_config.epochs}",
                "--ClientConfig.client_type", "TorchClient"
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
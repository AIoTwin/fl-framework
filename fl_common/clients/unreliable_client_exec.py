from spock import SpockBuilder

from data_retrieval import get_all_datasets
from fl_common import factory
from fl_common.train.trainer import ClientTrainer
from log_infra import build_wandb_metric_logger, def_logger, prepare_local_log_file
from misc.config_models import UnreliableClientConfig, DatasetsConfig, LoggingConfig, ModelZooConfig, ClientConfig
from misc.util import recursive_vars
from models.registry import load_model_from_zoo

logger = def_logger.getChild(__name__)

if __name__ == '__main__':
    config_space = SpockBuilder(UnreliableClientConfig,
                                LoggingConfig,
                                ModelZooConfig,
                                DatasetsConfig,
                                lazy=True,
                                desc="Build and start client").generate()
    logging_config = config_space.LoggingConfig
    zoo_config = config_space.ModelZooConfig
    unreliable_client_config = config_space.UnreliableClientConfig
    dataset_config = config_space.DatasetsConfig
    client_id = int(unreliable_client_config.client_id)
    prepare_local_log_file(log_file_path=logging_config.local_logging_config.log_file_path,
                           mode="a",
                           overwrite=False)
    client_wandb_metric_logger = build_wandb_metric_logger(
        wandb_config=logging_config.wandb_config,
        experiment_config_to_log=recursive_vars(unreliable_client_config),
        run_postfix=f"-client={client_id}",
        defer_init=False,
    )
    model = load_model_from_zoo(device=unreliable_client_config.device, zoo_config=zoo_config)
    client_trainer = ClientTrainer(
        trainer_configuration=unreliable_client_config.trainer_config,
        metric_logger=client_wandb_metric_logger,
        device=unreliable_client_config.device,
        datasets_dict=get_all_datasets(datasets_config=dataset_config.params),
        client_id=client_id,
    )
    client = factory.create_unreliable_client(
        unreliable_client_config=unreliable_client_config,
        client_trainer=client_trainer,
        constructed_model=model,
        client_id=client_id,
        failure_rate=unreliable_client_config.failure_rate
    )
    client.start()

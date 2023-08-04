from typing import Any, Dict, List, Optional, Union

from spock import SpockBuilder, spock


# TODO: Get device from a single shared source

# Modeland Dataset classes are too diverse and constraining to statically type this
@spock
class DatasetsConfig:
    params: Dict[str, object]


@spock
class CentralTestConfig:
    central_dataset_id: str
    central_loader_params: Dict[str, object]
    central_log_freq: int = 1000
    central_eval_metrics: List[str]
    central_main_metric: str = "Accuracy"


@spock
class EvalConfig:
    eval_dataset_id: str
    eval_loader_params: Dict[str, object]
    log_freq: int = 1000
    eval_metrics: List[str]
    main_metric: str = "Accuracy"


@spock
class TrainConfig:
    train_dataset_id: str  # id of dataset in dataset_dict
    epochs: int
    train_loader_params: Dict[str, object]
    log_freq: int = 1000
    optimizer: Dict[str, object]
    scheduler: Optional[Dict[str, object]]
    criterion: Dict[str, object]
    max_grad_norm: Optional[float]
    grad_accum_steps: int = 1


@spock
class TrainerConfig:
    sampler_indices: Optional[List[int]]
    train_config: Optional[TrainConfig]
    validation_config: Optional[EvalConfig]
    test_config: Optional[EvalConfig]
    ckpt_path: Optional[str]


@spock
class StrategyParams:
    fraction_fit: int = 1
    fraction_evaluate: int = 1
    min_available_clients: int
    min_fit_clients: int
    min_evaluate_clients: int


@spock
class ServerParams:
    server_address: str
    rounds: int


@spock
class StrategyConfig:
    strategy_type: str = "FedAvg"
    strategy_params: StrategyParams


@spock
class ServerConfig:
    server_type: str
    device: str
    num_reliable_clients: int
    num_unreliable_clients: int
    server_params: ServerParams
    strategy_config: StrategyConfig
    central_test_config: CentralTestConfig


@spock
class ClientConfig:
    client_type: str
    client_id: Optional[str]  # override with spock cli when creating thread
    device: str
    server_address: Optional[str]  # override to dynamically asisigning server address (e.g., after clustering)
    trainer_config: TrainerConfig
    client_params: Dict[str, object] = dict()


@spock
class UnreliableClientConfig():
    unreliable_client_type: str
    failure_rate: Optional[int]
    client_id: Optional[str]  # override with spock cli when creating thread
    device: str
    server_address: Optional[str]  # override to dynamically asisigning server address (e.g., after clustering)
    trainer_config: TrainerConfig
    client_params: Dict[str, object] = dict()


@spock
class AggregatorConfig:
    """
        As intended right now, aggregators will be the abstraction we need to support clustering.
        Each aggregator associates a client type to a server type. Although clusters consist of heterogenous client
        nodes, the clusters can be heterogenous among themselves.

        We can think about extending this if we require heterogeneity in client types within a cluster.
    """
    trainer_base_config: TrainerConfig
    server_config: ServerConfig
    client_config: ClientConfig


@spock
class WandBConfig:
    project: Optional[str]
    entity: Optional[str]
    notes: Optional[str]
    tags: Optional[List[str]]
    group: Optional[str]
    enabled: bool = False
    resume: bool = False
    id: Optional[str]
    run_name: str
    scalar_freq: int = 1000



@spock
class LocalLoggingConfig:
    config_path: Optional[str]
    log_file_path: Optional[str]
    overwrite: bool = False
    start_epoch: int = 0

    def __post_hook__(self):
        assert self.config_path or self.log_file_path, "Set either config or log file path"


@spock
class LoggingConfig:
    wandb_config: WandBConfig
    local_logging_config: LocalLoggingConfig


@spock
class ModelZooConfig:
    zoo_or_custom_model_name: str
    model_args: Dict[str, object]
    repo_or_dir: Optional[str]
    ckpt: str = "skip"


@spock
class BaseExperimentConfig:
    device: str
    test_only: bool = False
    model_config: ModelZooConfig
    logging_config: Optional[LoggingConfig]
    aggregator_config: Optional[AggregatorConfig]
    # it is better to build dataset at the "root" of the program, especially when we want to create a hierarchical
    #  dataset, since then the clients will pick a subset from a subset that was assigned to an aggregator.
    datasets: DatasetsConfig

    # def __post_hook__(self):
    #     assert not (
    #             self.aggregator_config.trainer_base_config.test_config is None and self.test_only
    #     ), "test configuration is necessary if test_only"

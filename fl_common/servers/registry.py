import copy
import functools
import flwr as fl
import numpy as np
from flwr.common.typing import Scalar, NDArrays
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from flwr.server.strategy import Strategy
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data_retrieval.retrieval import build_data_loader
from fl_common.process_handler import ThreadWrapper
from fl_common.train.eval_metrics import get_eval_metrics
from log_infra import WandBMetricLogger, def_logger
from misc.config_models import CentralTestConfig
from misc.util import ndarray_to_weight_dict

_SERVER_REGISTRY = dict()

logger = def_logger.getChild(__name__)


class IFlowerServer(ABC):
    @property
    @abstractmethod
    def server_address(self) -> str:
        raise NotImplementedError

    @server_address.setter
    @abstractmethod
    def server_address(self, server_address: str):
        raise NotImplementedError

    @abstractmethod
    def start(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError


def register_flower_server(_func: IFlowerServer = None, *, name: Optional[str] = None):
    def decorator_register(cls):
        @functools.wraps(cls)
        def wrapper_register():
            cls_name = name or cls.__name__
            _SERVER_REGISTRY[cls_name] = cls
            return cls

        return wrapper_register()

    if _func is None:
        return decorator_register
    else:
        return decorator_register(_func)


@register_flower_server
class TorchServer(IFlowerServer):
    def __init__(
        self,
        server_address: str,
        rounds: int,
        strategy: Callable[..., Strategy],
        *args,
        **kwargs,
    ):
        self.server_address = server_address
        self.rounds = rounds
        self.strategy = strategy()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def server_address(self) -> str:
        return self._server_address

    @server_address.setter
    def server_address(self, server_address: str):
        self._server_address = server_address

    def start(self):
        logger.info(f"Starting server at address {self.server_address}...")
        fl.server.start_server(
            server_address=self.server_address,
            config=fl.server.ServerConfig(num_rounds=self.rounds),
            strategy=self.strategy,
        )

    def stop(self):
        raise NotImplementedError("How do you even stop a flwr server?")


# todo: Replace with passing eval fn to Aggregator (sharable with proxy client)
@register_flower_server
class TorchServerWithCentralizedEval(TorchServer):
    def __init__(
        self,
        server_address: str,
        rounds: int,
        strategy: Callable[..., Strategy],
        model: nn.Module,
        device: str,
        wandb_metric_logger: WandBMetricLogger,
        test_config: CentralTestConfig,
        test_set: Dataset,
        *args,
        **kwargs,
    ):
        super().__init__(server_address, rounds, strategy, *args, **kwargs)
        self.strategy = strategy(evaluate_fn=self.centralized_eval)

        self.device = device
        self.model = model
        self.test_config = test_config
        self.wandb_metric_logger = wandb_metric_logger
        self.loader = build_data_loader(
            dataset=test_set, data_loader_config=self.test_config.central_loader_params
        )

    def centralized_eval(
        self, server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if hasattr(self, "parent_address"):
            log_str = f"Evaluating on local aggregator @{self.server_address} with parent @{self.parent_address}"
        else:
            log_str = f"Evaluating on global aggregator @{self.server_address}"
        log_str = (
            f"Evaluating before training...{log_str}"
            if server_round == 0
            else f"Finishing round {server_round}/{self.rounds}:{log_str} "
        )
        logger.info(log_str)

        eval_metrics = get_eval_metrics(
            metric_names=self.test_config.central_eval_metrics
        )
        # todo: Iterate through all eval metrics and then get results to return with the main_metric key
        eval_metric = eval_metrics.get(self.test_config.central_main_metric)

        self.model.to(self.device)
        state_dict = ndarray_to_weight_dict(self.model.state_dict().keys(), parameters)
        self.model.load_state_dict(state_dict, strict=False)

        name = self.test_config.central_main_metric
        prefix = "testing"
        result_dict = eval_metric.eval_func(
            model=self.model,
            data_loader=self.loader,
            device=self.device,
            title=f"{prefix.capitalize()} {name}:",
            header="Server",
        )
        self.wandb_metric_logger.wandblogger.log(
            {f"{prefix}/{k}" if k != "epoch" else k: v for k, v in result_dict.items()}
        )
        # todo: Do not assume result dict keys
        cross_entropy, accuracy = result_dict["Cross Entropy"], result_dict["acc@1"]
        # do we need to return loss as a single float??
        return cross_entropy, {"accuracy": accuracy}


# Todo: delete
@register_flower_server
class BidirectionalServer(TorchServer):
    @property
    def model(self):
        return self._client.runnable_fl_entity.model

    @model.setter
    def model(self, model):
        self._model = model

    class _AnonymousClient(fl.client.NumPyClient):
        def __init__(
            self,
            model: nn.Module,
            device: str,
            server_address: str,
            test_loader: DataLoader,
            eval_config: CentralTestConfig,
            metric_logger: WandBMetricLogger,
        ):
            self.model = model
            self.device = device
            self.server_address = server_address
            self.loader = test_loader
            self.eval_config = eval_config
            self.metric_logger = metric_logger

        def evaluate(
            self, parameters: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            eval_config = self.eval_config
            eval_metrics = get_eval_metrics(
                metric_names=eval_config.central_eval_metrics
            )
            eval_metric = eval_metrics.get(eval_config.central_main_metric)
            self.model.to(self.device)
            state_dict = ndarray_to_weight_dict(
                self.model.state_dict().keys(), parameters
            )
            self.model.load_state_dict(state_dict, strict=False)

            name = eval_config.central_main_metric
            prefix = "testing-local-aggregator"
            result_dict = eval_metric.eval_func(
                model=self.model,
                data_loader=self.loader,
                device=self.device,
                title=f"{prefix.capitalize()} {name}:",
                header="Testing at Local Aggregator",
            )
            self.metric_logger.wandblogger.log(
                {
                    f"{prefix}/{k}" if k != "epoch" else k: v
                    for k, v in result_dict.items()
                }
            )
            cross_entropy, accuracy = result_dict["Cross Entropy"], result_dict["acc@1"]
            # do we need to return loss as a single float??
            return (
                float(cross_entropy),
                len(self.loader.dataset),
                {"accuracy": float(accuracy)},
            )

        def fit(
            self, parameters, *args, **kwargs
        ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
            self.set_parameters(parameters)
            # todo: Figure out whether I need to return set size here
            return self.get_parameters(config={}), 0, {}

        def get_parameters(self, config) -> List[np.ndarray]:
            self.model.train()
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        def set_parameters(self, parameters: List[np.ndarray]) -> None:
            state_dict = ndarray_to_weight_dict(
                self.model.state_dict().keys(), parameters
            )
            self.model.load_state_dict(state_dict, strict=False)

        def start(self, *args, **kwargs):
            logger.info(
                f"Connecting child aggregator to parent at {self.server_address}"
            )
            fl.client.start_numpy_client(
                server_address=self.server_address, client=self
            )

    def __init__(
        self,
        model: nn.Module,
        device: str,
        wandb_metric_logger: WandBMetricLogger,
        test_config: CentralTestConfig,
        server_address: str,
        rounds: int,
        test_set: Dataset,
        strategy: Strategy,
        parent_address: str,
        *args,
        **kwargs,
    ):
        super().__init__(
            model,
            device,
            wandb_metric_logger,
            test_config,
            server_address,
            rounds,
            test_set,
            strategy,
        )
        self.parent_address = parent_address
        self._client = ThreadWrapper(
            self._AnonymousClient(
                model=model,
                device=self.device,
                server_address=parent_address,
                test_loader=self.loader,
                eval_config=self.test_config,
                metric_logger=self.wandb_metric_logger,
            )
        )

    def start(self):
        self.wandb_metric_logger.wandblogger.init()
        self._client.run()
        logger.info(
            f"Starting mid-level aggregator server at address {self.server_address}..."
        )
        fl.server.start_server(
            server_address=self.server_address,
            config=fl.server.ServerConfig(num_rounds=self.rounds),
            strategy=self.strategy,
        )


def get_flower_server(name: str) -> Callable[..., TorchServer]:
    cls = _SERVER_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Server class with name `{name}` not registered")
    return cls

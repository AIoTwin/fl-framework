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
from fl_common.train.eval_metrics import get_eval_metrics
from log_infra import WandBMetricLogger, def_logger
from misc.config_models import CentralTestConfig, ServerConfig
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
            model: nn.Module,
            device: str,
            wandb_metric_logger: WandBMetricLogger,
            test_config: CentralTestConfig,
            server_address: str,
            rounds: int,
            test_set: Dataset,
            strategy: Strategy,
            *args,
            **kwargs
    ):
        self.device = device
        self.model = model
        self.test_config = test_config
        self.wandb_metric_logger = wandb_metric_logger
        self.server_address = server_address
        self.rounds = rounds
        self.strategy = strategy
        self.loader = build_data_loader(
            dataset=test_set,
            data_loader_config=self.test_config.central_loader_params
        )

    # note: if I can't pass as a class method, use static method and partial
    def evaluate(self,
                 server_round: int,
                 parameters: NDArrays,
                 config: Dict[str, Scalar]
                 ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        logger.info(f"Evaluating on Central Server for round #{server_round}")

        eval_metrics = get_eval_metrics(metric_names=self.test_config.central_eval_metrics)
        # todo: Iterate through all eval metrics and then get results to return with the main_metric key
        eval_metric = eval_metrics.get(self.test_config.central_main_metric)

        self.model = copy.deepcopy(self.model)
        self.model.to(self.device)
        state_dict = ndarray_to_weight_dict(
            self.model.state_dict().keys(), parameters
        )
        self.model.load_state_dict(state_dict, strict=False)

        name = self.test_config.central_main_metric
        prefix = "testing"
        result_dict = eval_metric.eval_func(
            model=self.model,
            data_loader=self.loader,
            device=self.device,
            title=f"{prefix.capitalize()} {name}:",
            header="Centralized Test",
        )
        self.wandb_metric_logger.wandblogger.log(
            {f"{prefix}/{k}" if k != "epoch" else k: v for k, v in result_dict.items()}
        )
        # todo: Do not assume result dict keys
        cross_entropy, accuracy = result_dict["Cross Entropy"], result_dict["acc@1"]
        # do we need to return loss as a single float??
        return cross_entropy, {"accuracy": accuracy}

    @property
    def server_address(self) -> str:
        return self._server_address

    @server_address.setter
    def server_address(self, server_address: str):
        self._server_address = server_address

    def start(self):
        self.wandb_metric_logger.wandblogger.init()

        logger.info(f"Starting server at address {self.server_address}...")
        fl.server.start_server(
            server_address=self.server_address,
            config=fl.server.ServerConfig(num_rounds=self.rounds),
            strategy=self.strategy
        )

    def stop(self):
        raise NotImplementedError("How do you even stop a flwr server?")


@register_flower_server
class BidirectionalServer(TorchServer):
    class _AnonymousClient(fl.client.NumPyClient):
        def __init__(self,
                     model: nn.Module,
                     device: str,
                     server_address: str,
                     test_loader: DataLoader,
                     eval_config: CentralTestConfig,
                     metric_logger: WandBMetricLogger):
            self.model = model
            self.device = device
            self.server_address = server_address
            self.loader = test_loader
            self.eval_config = eval_config
            self.metric_logger = metric_logger

        def evaluate(
                self,
                parameters,
                *args,
                **kwargs
        ) -> Tuple[float, int, Dict[str, Any]]:
            self.set_parameters(parameters)
            loss = kwargs.get("loss")
            accuracy = kwargs.get("accuracy")
            set_size = kwargs.get("set_size")
            return (
                float(loss),
                set_size,
                {"accuracy": float(accuracy)},
            )

        def evaluate(self,
                     parameters: NDArrays,
                     config: Dict[str, Scalar]
                     ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            eval_config = self.eval_config
            eval_metrics = get_eval_metrics(metric_names=eval_config.central_eval_metrics)
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
                {f"{prefix}/{k}" if k != "epoch" else k: v for k, v in result_dict.items()}
            )
            cross_entropy, accuracy = result_dict["Cross Entropy"], result_dict["acc@1"]
            # do we need to return loss as a single float??
            return cross_entropy, {"accuracy": accuracy}

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
            state_dict = ndarray_to_weight_dict(self.model.state_dict().keys(), parameters)
            self.model.load_state_dict(state_dict, strict=False)

        def start(self, *args, **kwargs):
            logger.info(f"Connecting child aggregator to parent at {self.server_address}")
            fl.client.start_numpy_client(server_address=self.server_address, client=self)

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
            **kwargs
    ):
        super().__init__(model,
                         device,
                         wandb_metric_logger,
                         test_config,
                         server_address,
                         rounds,
                         test_set,
                         strategy)
        self._client = self._AnonymousClient(model=model,
                                             device=self.device,
                                             server_address=parent_address,
                                             test_loader=self.loader,
                                             eval_config=self.test_config,
                                             metric_logger=self.wandb_metric_logger)

    def start(self):
        self.wandb_metric_logger.wandblogger.init()

        logger.info(f"Starting server at address {self.server_address}...")
        fl.server.start_server(
            server_address=self.server_address,
            config=fl.server.ServerConfig(num_rounds=self.rounds),
            strategy=self.strategy
        )


def get_flower_server(name: str) -> Callable[..., TorchServer]:
    cls = _SERVER_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Server class with name `{name}` not registered")
    return cls

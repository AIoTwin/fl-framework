import functools
import flwr as fl
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from flwr.client import NumPyClient
from overrides import overrides
from torch import nn

from fl_common.train.trainer import ClientTrainer
from log_infra import def_logger
from misc.util import ndarray_to_weight_dict


_CLIENT_REGISTRY = dict()

logger = def_logger.getChild(__name__)


class TorchBaseClient(NumPyClient):
    """
    Base class with common parameter loading and connection logic
    """

    def __init__(
        self,
        model: nn.Module,
        server_address: str,
        set_sizes: Optional[Dict[str, int]] = None,
        *args,
        **kwargs,
    ):
        self.server_address = server_address
        self.model = model
        self.set_sizes = set_sizes
        # Client should have control to select its subset of data

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        self.model.train()
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        state_dict = ndarray_to_weight_dict(self.model.state_dict().keys(), parameters)
        self.model.load_state_dict(state_dict, strict=True)

    def start(self, log_str: Optional[str] = None):
        if log_str is None:
            log_str = f"Starting client connecting to {self.server_address}"
        logger.info(log_str)
        # todo: Move metric logger to client and pass it to Trainer
        # self.trainer.metric_logger.wandblogger.init()
        fl.client.start_numpy_client(server_address=self.server_address, client=self)


def register_flower_client(
    _func: TorchBaseClient = None, *, name: Optional[str] = None
):
    def decorator_register(cls: TorchBaseClient):
        @functools.wraps(cls)
        def wrapper_register():
            cls_name = name or cls.__name__
            _CLIENT_REGISTRY[cls_name] = cls
            return cls

        return wrapper_register()

    if _func is None:
        return decorator_register
    else:
        return decorator_register(_func)


@register_flower_client(name="TorchClient")
class TorchClient(TorchBaseClient):
    def __init__(
        self,
        model: nn.Module,
        server_address: str,
        client_id: str,
        client_trainer: ClientTrainer,
        monitor: bool = False,  # ???
        *args,
        **kwargs,
    ):
        super().__init__(model, server_address, *args, **kwargs)
        logger.debug(f"Constructing client with id: {client_id}")
        self.client_id = client_id
        # Client should have control to select its subset of data
        self.trainer = client_trainer
        self.monitor = monitor
        self.set_sizes = client_trainer.set_sizes

    def fit(
        self, parameters: List[np.array], config: Dict[str, Any], *args, **kwargs
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        :parameters: model weights as list of numpy arrays
        :config: arg set by Flower, passed by on_fit_config_fn parameter of Strategy
        """
        self.set_parameters(parameters)
        self.trainer.train(self.model)
        return self.get_parameters(config={}), self.set_sizes["train"], {}

    def evaluate(
        self, parameters: List[np.array], config: Dict[str, Any], *args, **kwargs
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        :parameters: model weights as list of numpy arrays
        :config: arg set by Flower, passed by on_evaluate_config parameter of Strategy
        """
        self.set_parameters(parameters)
        result_dict = self.trainer.validate(self.model)
        # todo: Don't assume validation metrics
        loss, accuracy = result_dict["Cross Entropy"], result_dict["acc@1"]
        return (
            float(loss),
            self.trainer.set_sizes["test"],
            {"accuracy": float(accuracy)},
        )

    def start(self, log_str: Optional[str] = None):
        if log_str is None:
            log_str = f"Starting client with id {self.client_id} connecting to server at {self.server_address}"
        super().start(log_str=log_str)


@register_flower_client(name="UnreliableClient")
class UnreliableTorchClient(TorchClient):
    def __init__(
        self,
        model: nn.Module,
        client_id: str,
        client_trainer: ClientTrainer,
        fail_at_round: int,
        server_address: str,
        *args,
        **kwargs,
    ):
        logger.debug(f"Constructing unreliable client with id: {client_id}")
        self.server_address = server_address
        self.client_id = client_id
        self.model = model
        self.trainer = client_trainer
        self.fail_at_round = fail_at_round * self.trainer.epochs
        print(self.fail_at_round)
        self.count = 0

    @overrides
    def fit(
        self, parameters, *args, **kwargs
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        self.count = self.count + 1
        if self.count % self.fail_at_round == 0:
            logger.info(
                f"Client with id {self.client_id} failed - no training this round!"
            )
        else:
            self.set_parameters(parameters)
            self.trainer.train(self.model)
            return self.get_parameters(config={}), self.trainer.set_sizes["train"], {}

    @overrides
    def evaluate(
        self, parameters, *args, **kwargs
    ) -> Tuple[float, int, Dict[str, Any]]:
        if self.count % self.fail_at_round == 0:
            logger.info(
                f"Client with id {self.client_id} failed - no evaluation this round!"
            )
        else:
            self.set_parameters(parameters)
            result_dict = self.trainer.validate(self.model)
            loss, accuracy = result_dict["Cross Entropy"], result_dict["acc@1"]
            return (
                float(loss),
                self.trainer.set_sizes["test"],
                {"accuracy": float(accuracy)},
            )

    def start(self, log_str: Optional[str] = None):
        if log_str is None:
            log_str = (
                f"Starting unreliable client (failure_rate={self.fail_at_round}) "
                f"with id {self.client_id} connecting to server at {self.server_address}"
            )
        super().start(log_str=log_str)


def get_flower_client(name: str) -> Callable[..., TorchClient]:
    cls = _CLIENT_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Client class with name `{name}` not registered")
    return cls

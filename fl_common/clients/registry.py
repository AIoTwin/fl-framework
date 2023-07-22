import functools
import flwr as fl
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from flwr.client import NumPyClient
from torch import nn

from fl_common.train.trainer import ClientTrainer
from log_infra import def_logger
from misc.util import ndarray_to_weight_dict


class FlowerBaseClient(NumPyClient):
    def start(self, *args, **kwargs):
        raise NotImplementedError


_CLIENT_REGISTRY = dict()

logger = def_logger.getChild(__name__)


def register_flower_client(_func: Callable = None, *, name: Optional[str] = None):
    def decorator_register(cls: FlowerBaseClient):
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
class TorchClient(FlowerBaseClient):
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
        logger.debug(f"Constructing client with id: {client_id}")
        self.server_address = server_address
        self.client_id = client_id
        self.model = model
        # Client should have control to select its subset of data
        self.trainer = client_trainer
        self.monitor = monitor

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        self.model.train()
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        state_dict = ndarray_to_weight_dict(self.model.state_dict().keys(), parameters)
        self.model.load_state_dict(state_dict, strict=False)

    def fit(
            self, parameters, *args, **kwargs
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        self.set_parameters(parameters)
        self.trainer.train(self.model)
        return self.get_parameters(config={}), self.trainer.set_sizes["train"], {}

    def evaluate(
            self, parameters, *args, **kwargs
    ) -> Tuple[float, int, Dict[str, Any]]:
        self.set_parameters(parameters)
        result_dict = self.trainer.validate(self.model)
        # todo: Don't assume validation metrics
        loss, accuracy = result_dict["Cross Entropy"], result_dict["acc@1"]
        return (
            float(loss),
            self.trainer.set_sizes["test"],
            {"accuracy": float(accuracy)},
        )

    # @staticmethod
    # def start(client):
    #     logger.info(f"Starting client with id {client.client_id}")
    #     client.trainer.metric_logger.wandblogger.init()
    #     fl.client.start_numpy_client(server_address=client.server_address, client=client)

    def start(self):
        logger.info(f"Starting client with id {self.client_id}")
        self.trainer.metric_logger.wandblogger.init()
        fl.client.start_numpy_client(server_address=self.server_address, client=self)


@register_flower_client
class FailableTorchClient(TorchClient):
    def __init__(
            self,
            model: nn.Module,
            monitor: bool,
            client_id: str,
            client_trainer: ClientTrainer,
            failure: int,
            *args,
            **kwargs,
    ):
        super().__init__(model, monitor, client_id, client_trainer, *args, **kwargs)
        self.failure = failure
        self.count = 0

    def fit(
            self, parameters, *args, **kwargs
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        if self.failure != 0:
            self.count = self.count + 1
            if self.count % self.failure == 0:
                raise Exception("Client failed!")
        self.set_parameters(parameters)
        self.trainer.train(self.model)
        return self.get_parameters(config={}), self.trainer.set_sizes["train"], {}


def get_flower_client(name: str) -> Callable[..., TorchClient]:
    cls = _CLIENT_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Client class with name `{name}` not registered")
    return cls

import copy
import functools
import flwr as fl
from flwr.common.typing import Scalar, NDArrays
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from data_retrieval.retrieval import build_data_loader
from fl_common.train.eval_metrics import get_eval_metrics
from log_infra import WandBMetricLogger, def_logger
from misc.config_models import ServerConfig, ServerParams
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
            server_config: ServerConfig,
            wandb_metric_logger: WandBMetricLogger,
            test_set: Dataset,
    ):
        self.model = model
        self.server_config = server_config
        self.device = server_config.device
        self.server_address = self.server_config.server_params.server_address
        self.wandb_metric_logger = wandb_metric_logger
        self.loader = build_data_loader(
            dataset=test_set,
            data_loader_config=server_config.central_test_config.central_loader_params
        )
        # self.loader = DatasetContainer.get_loader(self.server_config.central_test_config.central_dataset_id,
        #                                           self.server_config.central_test_config.central_loader_params)

    # note: if I can't pass as a class method, use static method and partial
    def evaluate(self,
                 server_round: int,
                 parameters: NDArrays, config: Dict[str, Scalar]
                 ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        logger.info(f"Evaluating on Central Server for round #{server_round}")
        eval_config = self.server_config.central_test_config
        eval_metrics = get_eval_metrics(metric_names=eval_config.central_eval_metrics)
        # todo: Iterate through all eval metrics and then get results to return with the main_metric key
        eval_metric = eval_metrics.get(eval_config.central_main_metric)

        self.model = copy.deepcopy(self.model)
        self.model.to(self.device)
        state_dict = ndarray_to_weight_dict(
            self.model.state_dict().keys(), parameters
        )
        self.model.load_state_dict(state_dict, strict=False)

        name = eval_config.central_main_metric
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

    # return evaluate

    @property
    def server_address(self) -> str:
        return self._server_address

    @server_address.setter
    def server_address(self, server_address: str):
        self._server_address = server_address

    def start(self):
        self.wandb_metric_logger.wandblogger.init()

        # todo: strategy registry and retrieve it according to strategy_type
        strategy = fl.server.strategy.FedAvg(
            **vars(self.server_config.strategy_config.strategy_params),
            evaluate_fn=self.evaluate
        )

        logger.info(f"Starting server at address {self.server_address}...")
        fl.server.start_server(
            server_address=self.server_address,
            config=fl.server.ServerConfig(num_rounds=self.server_config.server_params.rounds),
            strategy=strategy
        )

    def stop(self):
        raise NotImplementedError("How do you even stop a flwr server?")


class HierarchicalServer(TorchServer):
    """
        def __init__(children, parent, ...)

        @property
        def is_root(self):
            return self.parent is None
        ...
    """


def get_flower_server(name: str) -> Callable[..., TorchServer]:
    cls = _SERVER_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Server class with name `{name}` not registered")
    return cls

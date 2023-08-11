# TODO
#  Aggregator is an abstraction layer above Server to run as a thread managed by a ThreadExecutor
import concurrent.futures
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from flwr.common import NDArrays, Scalar

import flwr as fl
from torch.utils.data import Dataset

from data_retrieval.retrieval import build_data_loader
from fl_common.servers.registry import IFlowerServer
from fl_common.process_handler import ThreadWrapper
from fl_common.train.eval_metrics import get_eval_metrics
from log_infra import WandBMetricLogger, def_logger
from misc.config_models import CentralTestConfig
from misc.util import IterableSimpleNamespace, ndarray_to_weight_dict


logger = def_logger.getChild(__name__)


class AggregatorParentConnection(fl.client.NumPyClient):
    def __init__(
        self,
        device: str,
        parent_address: str,
        test_set: Dataset,
        eval_config: CentralTestConfig,
        metric_logger: WandBMetricLogger,
        shared_state: IterableSimpleNamespace,
        # todo: pass as (shared) trainer between agg parent connection and server together with eval(_config)
        aggregator_fit_fn: Optional[Callable] = None,
    ):
        self.device = device
        self.parent_address = parent_address
        self.eval_config = eval_config
        self.loader = build_data_loader(
            dataset=test_set, data_loader_config=self.eval_config.central_loader_params
        )
        self.metric_logger = metric_logger
        self._shared_state = shared_state

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        logger.info("Halting child aggregator")
        self._shared_state.server_lock.acquire()
        logger.info(
            f"Child aggregator sending parameters to parent aggregator @ {self.parent_address} "
            f"for round {self._shared_state.current_local_round}"
        )
        return self.get_parameters(config={}), self._shared_state.local_samples_processed, {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        eval_config = self.eval_config
        eval_metrics = get_eval_metrics(metric_names=eval_config.central_eval_metrics)
        eval_metric = eval_metrics.get(eval_config.central_main_metric)
        self._shared_state.model.to(self.device)
        state_dict = ndarray_to_weight_dict(self._shared_state.model.state_dict().keys(), parameters)

        logger.info(
            "Child aggregator - Releasing lock for children  to continue..."
        )

        self._shared_state.client_lock.release()

        self._shared_state.model.load_state_dict(state_dict, strict=False)

        name = eval_config.central_main_metric
        prefix = "testing-local-aggregator"
        # result_dict = eval_metric.eval_func(
        #     model=self._shared_state.model,
        #     data_loader=self.loader,
        #     device=self.device,
        #     title=f"{prefix.capitalize()} {name}:",
        #     header="Centralized Testing at Local Aggregator",
        # )
        # self.metric_logger.wandblogger.log(
        #     {f"{prefix}/{k}" if k != "epoch" else k: v for k, v in result_dict.items()}
        # )
        # cross_entropy, accuracy = result_dict["Cross Entropy"], result_dict["acc@1"]
        cross_entropy, accuracy = 0, 0
        # do we need to return loss as a single float??
        return float(cross_entropy), len(self.loader.dataset), {"accuracy": float(accuracy)}

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self._shared_state.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        state_dict = ndarray_to_weight_dict(self._shared_state.model.state_dict().keys(), parameters)
        self._shared_state.model.load_state_dict(state_dict, strict=False)

    def start(self, *args, **kwargs):
        logger.info(f"Connecting child aggregator to parent @ {self.parent_address}")
        fl.client.start_numpy_client(server_address=self.parent_address, client=self)


class Aggregator:
    def __init__(
        self,
        server: IFlowerServer,
        parent_connection: Optional[AggregatorParentConnection],
    ):
        self._parent_connection = parent_connection
        self._server = server

    def start(self):
        if self._parent_connection:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                connection_future = executor.submit(
                    ThreadWrapper(self._parent_connection).run
                )
                server_future = executor.submit(ThreadWrapper(self._server).run)
                # try:
                #     result_parent = connection_future.result()
                #     logger.debug(result_parent)
                # except Exception as e:
                #     logger.warning("Parent exception:", e)
                # try:
                #     result_server = server_future.result()
                #     logger.debug(result_server)
                # except Exception as e:
                #     logger.warning("Aggregator Server exception:", e)
        else:
            self._server.start()

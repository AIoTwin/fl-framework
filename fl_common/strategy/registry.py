import logging
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.grpc_server.grpc_bridge import GrpcBridgeClosed
from flwr.server.strategy import Strategy

from misc.util import IterableSimpleNamespace, ndarray_to_weight_dict

logger = logging.getLogger(__name__)

_STRATEGY_REGISTRY = {
    strategy.__name__: strategy for strategy in Strategy.__subclasses__()
}


class StrategyWithLockingWrapper(Strategy):
    def __init__(
            self,
            strategy_cls: Callable[..., Strategy],
            *,
            shared_state: IterableSimpleNamespace,
            **strategy_kwargs,
    ):
        super().__init__()
        self._strategy = strategy_cls(**strategy_kwargs)
        self._shared_state = shared_state

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        new_fails = []
        new_results = []  # Initialize a new list to store filtered results
        for x in range(len(results)):
            if results[x][1].num_examples == -1:
                # If num_examples is -1, set the corresponding failure entry to True
                new_fails.append(GrpcBridgeClosed())
            else:
                new_results.append(results[x])
        parameters_aggregated, metrics_aggregated = self._strategy.aggregate_fit(
            server_round, new_results, new_fails
        )

        if parameters_aggregated:
            self._shared_state.model.load_state_dict(
                ndarray_to_weight_dict(self._shared_state.model.state_dict().keys(),
                                       parameters_to_ndarrays(parameters_aggregated))
            )
            local_samples_processed = 0
            for _, fit_res in new_results:
                local_samples_processed += fit_res.num_examples
            self._shared_state.local_samples_processed = local_samples_processed

        self._shared_state.current_local_round += 1

        logger.info(
            f"ServerThread - Finished local aggregation round: {self._shared_state.current_local_round}"
        )

        if self._shared_state.current_local_round == self._shared_state.no_local_rounds:
            self._shared_state.current_local_round = 0

            logger.info("Local aggregator releasing server lock")
            self._shared_state.server_lock.release()

            logger.info("Local aggregator acquiring client lock")
            self._shared_state.client_lock.acquire()

            logger.info(
                "Local aggregator Acquired client lock - Continuing locale round..."
            )

        return parameters_aggregated, metrics_aggregated

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return self._strategy.initialize_parameters(client_manager)

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        return self._strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        self._strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        self._strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        self._strategy.evaluate(server_round, parameters)


def _default_strategy_params(params_dict):
    num_children = params_dict["num_children"]
    del params_dict["num_children"]
    if "fraction_fit" not in params_dict:
        params_dict["fraction_fit"] = 1.0
    if "fraction_evaluate" not in params_dict:
        params_dict["fraction_evaluate"] = 1.0
    if "min_available_clients" not in params_dict:
        params_dict["min_available_clients"] = num_children
    if "min_fit_clients" not in params_dict:
        params_dict["min_fit_clients"] = num_children
    if "min_evaluate_clients" not in params_dict:
        params_dict["min_evaluate_clients"] = num_children


def get_strategy(
        name: str, partial_params: Optional[Dict[str, Any]], wrap_with_locking: bool = False
) -> Callable[..., Strategy]:
    cls = _STRATEGY_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Strategy class with name `{name}` not registered")
    if partial_params:
        _default_strategy_params(partial_params)
        cls = partial(cls, **partial_params)
    if wrap_with_locking:
        logger.info(
            f"Wrapping strategy {name} with locking mechanism for local aggregator..."
        )
        cls = partial(StrategyWithLockingWrapper, strategy_cls=cls)
    return cls

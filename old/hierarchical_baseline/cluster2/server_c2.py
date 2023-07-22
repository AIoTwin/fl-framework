import logging
import os
import threading
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate

from old import train_util
from old.hierarchical_baseline.client import get_model, ndarray_to_weight_dict

edge_num_examples = 0
k_c = 2
current_edge_round = 0
next_cloud_round = 2

cloud_aggregation_lock = threading.Lock()
cloud_aggregation_lock.acquire()

continue_edge_round_lock = threading.Lock()
continue_edge_round_lock.acquire()

model = get_model()

train_loader, _ = train_util.default_loaders("train")
test_loader, test_set_size = train_util.default_loaders("test")

DEVICE = os.environ.get("DEVICE", "cpu")

class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(logging.WARNING, "No fit_metrics_aggregation_fn provided")

        # custom added code to support Hierarchical FL
        global edge_num_examples
        global current_edge_round
        global next_cloud_round

        state_dict = ndarray_to_weight_dict(model.state_dict().keys(), parameters_aggregated)
        model.load_state_dict(state_dict, strict=True)

        edge_num_examples = 0
        for _, fit_res in results:
            edge_num_examples = edge_num_examples + fit_res.num_examples

        current_edge_round = current_edge_round + 1

        print("ServerThread - Finished edge aggregation: " + str(current_edge_round))

        if current_edge_round == next_cloud_round:
            next_cloud_round += k_c

            print("ServerThread - Releasing cloud aggregation lock...")
            cloud_aggregation_lock.release()

            print("ServerThread - Waiting for continue edge round lock...")
            continue_edge_round_lock.acquire()

            print("ServerThread - Continuing edge round...")

        return parameters_aggregated, metrics_aggregated


# Define Flower client
class EdgeServerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        global current_edge_round

        print("ClientThread - Waiting for cloud aggregation lock...")
        cloud_aggregation_lock.acquire()

        print("ClientThread - Sending parameters to cloud aggregation: " + str(current_edge_round))
        return model.get_weights(), edge_num_examples, {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)

        print("ClientThread - Releasing continue edge round lock...")
        continue_edge_round_lock.release()

        loss, accuracy = train_util.test(model, test_loader, device=DEVICE)
        return loss, test_set_size, {"accuracy": accuracy}


class clientThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        fl.client.start_numpy_client(server_address="127.0.0.1:5000", client=EdgeServerClient())


class serverThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        customStrategy = CustomFedAvg(
            fraction_fit=1,
            fraction_evaluate=1,
            min_available_clients=2,
            min_fit_clients=2,
            min_evaluate_clients=2,
            accept_failures=True
        )

        # Start Flower server
        fl.server.start_server(
            server_address="0.0.0.0:5002",
            config=fl.server.ServerConfig(num_rounds=50),
            strategy=customStrategy
        )


# Create new threads
thread1 = clientThread(1, "Client-Thread", 1)
thread2 = serverThread(2, "Server-Thread", 2)

# Start new Threads
thread1.start()
thread2.start()
thread1.join()
thread2.join()

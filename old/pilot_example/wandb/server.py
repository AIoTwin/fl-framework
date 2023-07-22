import argparse
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import wandb
from flwr.common.typing import NDArrays, Scalar

from utils import Net, load_test_data, test

# https://github.com/adap/flower/blob/main/examples/simulation_pytorch/main.py
parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

# For wandb
parser.add_argument("--number_clients", type=int, default=64)  # clients in the training
parser.add_argument("--project", type=str, default="")

# For the flower client
parser.add_argument("--id", type=str, default=0)
parser.add_argument("--rounds", type=int, default=0)

args = parser.parse_args()


# https://flower.dev/docs/evaluation.html
def get_evaluate_fn():
    """Return an evaluation function for centralized evaluation."""
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    test_loader = load_test_data()

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Server evaluate")
        model = Net()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

        loss, accuracy = test(model=model, test_loader=test_loader, device=device)
        print({"Test Loss": loss, "Test Accuracy": accuracy})
        wandb.log({"acc": accuracy, "loss": loss})
        return loss, {"accuracy": accuracy}

    return evaluate


if __name__ == "__main__":
    print("started server " + str(args.id))
    number_clients = args.number_clients

    wandb.init(
        project=args.project,
        name=f"server-{args.id}",
        mode="online",
    )
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_available_clients=number_clients,
        min_fit_clients=number_clients,
        min_evaluate_clients=number_clients,
        evaluate_fn=get_evaluate_fn(),
    )

    print("connect")

    # Start Flower server
    fl.server.start_server(
        server_address="127.0.0.1:5050",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

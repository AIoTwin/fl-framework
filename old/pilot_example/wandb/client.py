import argparse
from collections import OrderedDict

import flwr as fl
import torch
import wandb

from utils import Net, load_data, test, train

# https://github.com/adap/flower/blob/main/examples/simulation_pytorch/main.py
parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

# For wandb
parser.add_argument("--number_clients", type=int, default=64)  # clients in the training
parser.add_argument("--project", type=str, default="")

# For the flower client
parser.add_argument("--epochs", type=int, default=0)
parser.add_argument("--monitor", type=bool, default=False)
parser.add_argument("--id", type=str, default=0)
parser.add_argument("--failure", type=int, default=0, required=False)

args = parser.parse_args()

# Define Flower client
class CifarClient(fl.client.NumPyClient):

    def __init__(self,
                 train_loader,
                 test_loader,
                 num_examples, epochs, monitor,failure):
        self.device = torch.device("cpu")
        self.model = Net().to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_examples = num_examples
        self.epochs = epochs
        self.monitor = monitor
        self.failure = failure
        self.count=0

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        if self.failure!= 0:
            self.count = self.count + 1
            if self.count % self.failure == 0:
                raise Exception("Client failed!")
        self.set_parameters(parameters)
        train(self.model, self.train_loader, self.epochs, self.device)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.test_loader, self.device)
        wandb.log({"acc": accuracy, "loss": loss})

        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


if __name__ == "__main__":
    print("started client " + str(args.id))
    # https://docs.wandb.ai/guides/track/launch
    wandb.init(
        project=args.project,
        name=f"client-{args.number_clients}-{args.id}",
        mode="online",
    )
    id = int(args.id[1:])
    train_loader, test_loader, num_examples = load_data(id)

    # Start Flower client
    fl.client.start_numpy_client(server_address="127.0.0.1:5050",
                                 client=CifarClient(train_loader, test_loader, num_examples, args.epochs,
                                                    args.monitor,args.failure))

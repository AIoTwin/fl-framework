import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from old.train_util import default_loaders

DEVICE = os.environ.get("DEVICE", "cpu")
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

print("Training Images Shape (x train shape) :", train_images.shape)
print("Label of training images (y train shape) :", train_labels.shape)
print("Test Images Shape (x test shape) :", test_images.shape)
print("Label of test images (y test shape) :", test_labels.shape)

train_images, test_images = train_images / 255, test_images / 255

IMG_SHAPE = (32, 32, 3)

num_epochs = 10
fine_tune_epochs = 30
total_epochs = num_epochs + fine_tune_epochs


def _freeze_module_params(modules: Union[List[nn.Module], nn.Module]) -> None:
    modules = modules if isinstance(list, modules) else [modules]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False


def ndarray_to_weight_dict(keys: Iterable[str], params: List[np.ndarray]) -> Dict:
    params_dict = zip(keys, params)
    return OrderedDict({k: torch.tensor(v) for k, v in params_dict})


def get_model() -> nn.Module:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    _freeze_module_params(model.parameters())
    model.classifier = nn.Linear(in_features=model.last_channel, out_features=10)
    return model


class CifarClient(fl.client.NumPyClient):
    """
        Mostly from Tutorial https://flower.dev/docs/framework/example-pytorch-from-centralized-to-federated.html
    """

    def __init__(
            self,
            model: nn.Module,
            num_examples=None,
            train_loader: Optional[DataLoader] = None,
            test_loader: Optional[DataLoader] = None,
    ) -> None:
        if num_examples is None:
            num_examples = dict()
        self.model = model
        self.num_examples = num_examples
        if train_loader:
            self.train_loader, self.num_examples["train_set"] = default_loaders("train")
        if test_loader:
            self.test_loader, self.num_examples["test_set"] = default_loaders("test")
        else:
            self.test_loader = test_loader

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        state_dict = ndarray_to_weight_dict(self.model.state_dict().keys(), parameters)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
            self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        train_methods.train(self.model, self.train_loader, epochs=1, device=DEVICE)
        return self.get_parameters(config={}), self.num_examples["train_set"], {}

    def evaluate(
            self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = train_methods.test(self.model, self.test_loader, device=DEVICE)
        return float(loss), self.num_examples["test_set"], {"accuracy": float(accuracy)}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:5002", client=CifarClient())

from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.testing._internal.codegen.random_topo_test import DEVICE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size = 32
learning_rate = 0.001
num_epochs = 10
nr_of_samples = 521 * 32


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test(model, test_loader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def train(model, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def load_data():
    # Load and preprocess the CIFAR10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

    total_length = len(train_dataset)
    start_index = total_length - nr_of_samples
    indices = range(start_index, total_length)
    train_subset = torch.utils.data.Subset(train_dataset, indices)

    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    num_examples = {"trainset": len(train_subset), "testset": len(test_dataset)}
    print(num_examples)
    return train_loader, test_loader, num_examples


model = Net().to(DEVICE)
train_loader, test_loader, num_examples = load_data()


def get_criterion_and_optimizer():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return criterion,optimizer


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(model,train_loader, num_epochs)
        return self.get_parameters(config={}), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(model, test_loader)
        return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:5050", client=CifarClient())


from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from old import constants

"""
From tutorial

Implementation is very hacky and I'm not sure why they initialize a new optimizer each round.
Should be bad especially when using optimizers like Adam

"""

def train(
        net: nn.Module,
        trainloader: DataLoader,
        epochs: int,
        device: torch.device,
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


@torch.inference_mode()
def test(
        net: nn.Module,
        testloader: DataLoader,
        device: torch.device,
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def default_loaders(split: str) -> Tuple[DataLoader, int]:
    base_transforms = [transforms.Normalize(mean=constants.MEAN_CIFAR10,
                                            std=constants.STD_CIFAR10),
                       transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC)]
    if split == "train":
        split_set = datasets.CIFAR10(root="resources/datasets/cifar10/train",
                                     train=True,
                                     download=True,
                                     # add augmentation if wanted, e.g., transforms.RandomHorizontalFlip(p=0.5)
                                     transform=transforms.Compose(base_transforms))
        split_loader = DataLoader(dataset=split_set,
                                  batch_size=32,
                                  shuffle=True)
    elif split == "test":
        split_set = datasets.CIFAR10(root="resources/datasets/cifar10/test",
                                     train=False,
                                     download=True,
                                     # don't add any additional test time augmentation
                                     transform=transforms.Compose(base_transforms))
        split_loader = DataLoader(dataset=split_set,
                                  batch_size=128,
                                  shuffle=False)
    else:
        raise ValueError(f"Unknown split: {split}")
    return split_loader, len(split_set)

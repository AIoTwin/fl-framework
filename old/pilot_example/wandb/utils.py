import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# https://flower.dev/docs/quickstart-pytorch.html
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


def test(model, test_loader, device):
    print("TESTING")
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def train(net, trainloader, epochs, device):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        print("Training")
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def get_client_subset(train_dataset, id):
    num_classes = 10
    num_clients = 64

    class_indices = [[] for _ in range(num_classes)]

    for i, (_, label) in enumerate(train_dataset):
        class_indices[label].append(i)
    client_datasets = []

    # Split the class samples into subsets
    for part in range(num_clients):
        subset_indices = []

        # Iterate over class labels
        for class_label in range(num_classes):
            sample_range = round(len(class_indices[class_label]) / num_clients)
            start_index = round(part * sample_range)
            end_index = round((part + 1) * sample_range - 1)
            # Add the subset indices for the current class and part
            subset_indices.extend(class_indices[class_label][start_index:end_index])
        # Create the subset using the collected indices
        subset = Subset(train_dataset, subset_indices)
        client_datasets.append(subset)
    print("got subset")
    return client_datasets[id]


def load_test_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform)
    batch_size = 32
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def load_data(id):
    print("load data " + str(id))
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, transform=transform, download=True
    )
    train_data_subset = get_client_subset(train_dataset, id)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform)
    batch_size = 32
    train_loader = DataLoader(
        dataset=train_data_subset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    num_examples = {"trainset": len(train_data_subset), "testset": len(test_dataset)}
    return train_loader, test_loader, num_examples

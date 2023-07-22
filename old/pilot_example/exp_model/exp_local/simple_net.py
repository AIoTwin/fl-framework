import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


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
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Accuracy on test images: {(100 * correct / total)}%")
        append_to_file(
            "results.txt", f"Accuracy on test images: {(100 * correct / total)}%"
        )


def train(train_loader, num_epochs, device, model, criterion, optimizer):
    total_step = len(train_loader)
    loss_list = []
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(total_loss)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item()}"
        )


def append_to_file(file_path, content):
    try:
        with open(file_path, "a") as file:
            file.write(content)
            file.write("\n")
    except IOError:
        print("An error occurred while writing to the file.")


def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define training parameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    nr_of_samples = 521 * 32

    # Load and preprocess the CIFAR10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, transform=transform, download=True
    )
    train_subset = torch.utils.data.Subset(train_dataset, range(nr_of_samples))
    # Calculate the class weights
    subset_labels = torch.tensor([train_subset[i][1] for i in range(len(train_subset))])
    class_sample_counts = torch.bincount(subset_labels)
    class_weights = 1.0 / class_sample_counts.float()

    # Create a list of weights corresponding to each sample in the subset
    sample_weights = [class_weights[label] for label in subset_labels]

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    append_to_file("results.txt", "Simple Net")
    print(len(train_dataset))
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform)
    print(len(test_dataset))
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Create the CNN model
    model = Net().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(train_loader, num_epochs, device, model, criterion, optimizer)
    append_to_file(
        "results.txt", "Training time: " + str(round((time.time() - start_time), 2))
    )
    # Test the model
    test(model, test_loader, device)
    append_to_file("results.txt", "")


if __name__ == "__main__":
    main()

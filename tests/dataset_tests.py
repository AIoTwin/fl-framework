from collections import Counter
from math import floor

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets
import colorcet as cc


def sort_samples_per_class(n_classes, train_dataset):
    class_indices = [[] for _ in range(n_classes)]
    for i, (_, label) in enumerate(train_dataset):
        class_indices[label].append(i)
    return class_indices


def _flat_skewed_test(data_source: Dataset, n_classes, world_size,
                      rank) -> list:
    class_samples = sort_samples_per_class(n_classes, data_source)

    # Initialize the subsets for each client
    client_subsets = [[] for _ in range(world_size)]

    for class_nr, sublist in enumerate(class_samples):
        for i, sample_idx in enumerate(sublist):
            client_idx = i % world_size
            client_subsets[client_idx].append(
                (class_nr, sample_idx))  # different to code: save class here as well for simplification
    return client_subsets[rank]


# each client should only have one or two classes
def _flat_fair_test(data_source: Dataset,
                    n_classes,
                    world_size,
                    rank) -> list:
    class_samples = sort_samples_per_class(n_classes, data_source)
    nr_of_samples = len(data_source.targets)
    samples_per_client = np.ceil(nr_of_samples / world_size)
    client_subsets = [[] for _ in range(world_size)]
    sample_counter = 0
    for class_nr, sublist in enumerate(class_samples):
        for i, sample_idx in enumerate(sublist):
            client_idx = floor(sample_counter / samples_per_client)
            client_subsets[client_idx].append(
                (class_nr, sample_idx))  # different to code: save class here as well for simplification
            sample_counter += 1
    return client_subsets[rank]


def get_data():
    train_dataset = datasets.CIFAR10(root="resources/datasets/cifar10/train",
                                     train=True,
                                     download=True)
    return train_dataset


def count_samples_per_class(data):
    # Extract the first values from the tuples and count occurrences
    count_dict = Counter(item[0] for item in data)

    # Convert the Counter to a list of tuples
    count_list = list(count_dict.items())

    print("Data samples per class:", count_list)
    return count_list


def plot_sample_distribution(client_subsets, img_name, method):
    labels = list(set(item[0] for sublist in client_subsets for item in sublist))
    num_subsets = len(client_subsets)
    color_palette = sns.color_palette(cc.glasbey, n_colors=num_subsets)
    fig, ax = plt.subplots()

    # Create the stacked bar plot
    bottom = np.zeros(len(labels))
    for i, subset in enumerate(client_subsets):
        values = [0] * len(labels)
        for item in subset:
            values[labels.index(item[0])] = item[1]

        ax.bar(labels, values, color=color_palette[i], label=f'Client {i + 1}', bottom=bottom)
        bottom = np.add(bottom, values)

    ax.set_xlabel('Class id')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Client subsets for ' + str(num_subsets) + ' clients: ' + method)
    plt.show()
    #plt.savefig(img_name)


def get_all_client_data_flat_skewed_test(world_size):
    clients_samples_per_class = []
    for client_id in range(world_size):
        indices = _flat_skewed_test(train_dataset, 10, world_size, client_id)
        counted_samples = count_samples_per_class(indices)
        clients_samples_per_class.append(counted_samples)
    return clients_samples_per_class


def get_all_client_data_flat_fair_test(world_size):
    clients_samples_per_class = []
    for client_id in range(world_size):
        indices = _flat_fair_test(train_dataset, 10, world_size, client_id)
        counted_samples = count_samples_per_class(indices)
        clients_samples_per_class.append(counted_samples)
    return clients_samples_per_class


if __name__ == "__main__":
    train_dataset = get_data()
    clients_samples_per_class = get_all_client_data_flat_skewed_test(3)
    plot_sample_distribution(clients_samples_per_class, "flat_skewed_3_clients", "flat_skewed")
    clients_samples_per_class = get_all_client_data_flat_skewed_test(64)
    plot_sample_distribution(clients_samples_per_class, "flat_skewed_64_clients", "flat_skewed")
    clients_samples_per_class = get_all_client_data_flat_skewed_test(12)
    plot_sample_distribution(clients_samples_per_class, "flat_skewed_12_clients", "flat_skewed")
    clients_samples_per_class = get_all_client_data_flat_fair_test(64)
    plot_sample_distribution(clients_samples_per_class, "flat_fair_64_clients", "fair_skewed")
    clients_samples_per_class = get_all_client_data_flat_fair_test(3)
    plot_sample_distribution(clients_samples_per_class, "flat_fair_3_clients", "fair_skewed")
    clients_samples_per_class = get_all_client_data_flat_fair_test(12)
    plot_sample_distribution(clients_samples_per_class, "flat_fair_12_clients", "fair_skewed")

import logging
import random
from math import ceil, floor
from typing import Container, Dict, Optional

import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _flat_fair(data_source: Dataset,
               n_classes,
               world_size,
               rank) -> Dict[int, list]:
    logger.info(f"Building indices for subsets for client with rank {rank}..")
    class_samples = sort_samples_per_class(n_classes, data_source)
    nr_of_samples = len(data_source.targets)
    samples_per_client = ceil(nr_of_samples / world_size)
    client_subsets = [[] for _ in range(world_size)]
    sample_counter = 0
    for class_nr, sublist in enumerate(class_samples):
        for i, sample_idx in enumerate(sublist):
            client_idx = floor(sample_counter / samples_per_client)
            client_subsets[client_idx].append(sample_idx)
            sample_counter += 1
    current_subset = client_subsets[(int(world_size / n_classes) * rank) % world_size]
    random.shuffle(current_subset)
    return current_subset


def _flat_fair_2_classes_per_client(data_source: Dataset,
                                    n_classes,
                                    world_size,
                                    rank) -> Dict[int, list]:
    logger.info(f"Building indices for subsets for client with rank {rank}..")
    class_samples = sort_samples_per_class(n_classes, data_source)
    nr_of_samples = len(data_source.targets)
    samples_per_client = ceil(nr_of_samples / world_size)
    client_subsets = [[] for _ in range(world_size)]
    sample_counter = 0
    for class_nr, sublist in enumerate(class_samples):
        for i, sample_idx in enumerate(sublist):
            client_idx = floor(sample_counter / samples_per_client)
            client_subsets[client_idx].append(sample_idx)
            sample_counter += 1
    current_subset = client_subsets[(int(world_size / n_classes) * rank) % world_size] + client_subsets[
        ((int(world_size / n_classes) * (rank + 1)) + 1) % world_size]
    random.shuffle(current_subset)
    return current_subset


def sort_samples_per_class(n_classes, train_dataset):
    class_indices = [[] for _ in range(n_classes)]
    for i, (_, label) in enumerate(train_dataset):
        class_indices[label].append(i)
    return class_indices


def _flat_skewed(data_source: Dataset,
                 n_classes,
                 world_size,
                 rank) -> Dict[int, list]:
    logger.info(f"Building indices for subsets for client with rank {rank}..")
    class_samples = sort_samples_per_class(n_classes, data_source)

    client_subsets = [[] for _ in range(world_size)]

    for _, sublist in enumerate(class_samples):
        for i, sample_idx in enumerate(sublist):
            client_idx = i % world_size
            client_subsets[client_idx].append(sample_idx)

    logger.info(str(world_size))
    logger.info(str(rank))
    logger.info(f"Assigning {len(client_subsets[rank])} to client ith rank {rank}")
    random.shuffle(client_subsets[rank])
    return client_subsets[rank]


def build_indices(strategy: str, data_source: Dict[str, Dataset], rank: int, world_size: int, n_classes: int, *args,
                  **kwargs) -> Optional[int]:
    if strategy == "flat_fair":
        return _flat_fair(data_source, n_classes, world_size, rank, *args, **kwargs)
    elif strategy == "flat_skewed":
        return _flat_skewed(data_source, n_classes, world_size, rank, *args, **kwargs)
    elif strategy == "_flat_fair_2_classes":
        return _flat_fair_2_classes_per_client(data_source, n_classes, world_size, rank, *args, **kwargs)
    elif strategy == 'all_for_all':
        return None  # Won't use CustomSampler when indices are None

    raise ValueError(f"Strategy {strategy} not implemented")

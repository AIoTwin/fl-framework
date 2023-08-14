import logging
from typing import Container, Dict, Optional

logger = logging.getLogger(__name__)


def _flat_fair(data_source: Container,
               n_classes,
               world_size,
               rank) -> Dict[int, list]:
    logger.info(f"Building indices for subsets for client with rank {rank}..")
    # classes_per_worker = n_classes // world_size
    # index_dict = dict()
    # for rank in range(world_size):
    #     classes_to_include = set(range(rank * classes_per_worker, (rank + 1) * classes_per_worker))
    #
    #     indices = [i for i, sample in enumerate(data_source)
    #                if sample[1] in classes_to_include]
    #     index_dict[rank] = indices
    # return index_dict
    classes_per_worker = n_classes // world_size
    classes_to_include = set(range(rank * classes_per_worker, (rank + 1) * classes_per_worker))

    indices = [i for i, sample in enumerate(data_source)
               if sample[1] in classes_to_include]
    logger.info(f"Assigning {len(indices)} to client ith rank {rank}")
    return indices


def _flat_skewed(*args, **kwargs):
    # todo
    raise NotImplemented


def build_indices(strategy: str,
                  data_source: Container,
                  *args,
                  **kwargs) -> Optional[int]:
    if strategy == "flat_fair":
        return _flat_fair(data_source, *args, **kwargs)
    elif strategy == "flat_skewed":
        return _flat_skewed(*args, **kwargs)
    elif strategy == 'all_for_all':
        return None  # Won't use CustomSampler when indices are None

    raise ValueError(f"Strategy {strategy} not implemented")

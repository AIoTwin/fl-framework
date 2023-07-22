import time
from typing import Any, Dict, Iterable, Optional

import torchvision
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from torch.utils.data.distributed import DistributedSampler

from data_retrieval.collator import get_collate_func
from data_retrieval.preparation import get_torchvision_dataset, split_dataset
from data_retrieval.registry import DATASET_DICT
from data_retrieval.samplers.registry import CustomIndicesSampler, get_batch_sampler
from data_retrieval.wrapper import BaseDatasetWrapper, get_dataset_wrapper
from log_infra import def_logger

logger = def_logger.getChild(__name__)


def build_data_loader(
        dataset: Dataset,
        data_loader_config: Dict[str, Any],
        distributed: bool = False,
        sampler_indices: Optional[Iterable[int]] = None,
):
    num_workers = data_loader_config["num_workers"]
    dataset_wrapper_config = data_loader_config.get("dataset_wrapper", None)
    if isinstance(dataset_wrapper_config, dict) and len(dataset_wrapper_config) > 0:
        dataset = get_dataset_wrapper(
            dataset_wrapper_config["name"],
            dataset,
            **dataset_wrapper_config.get("params", dict())
        )
    elif data_loader_config.get("requires_supp", False):
        dataset = BaseDatasetWrapper(dataset)

    if sampler_indices is not None:
        sampler = CustomIndicesSampler(sampler_indices)
    elif distributed:
        sampler = DistributedSampler(dataset)
    elif data_loader_config.get("random_sample", False):
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    batch_sampler_config = data_loader_config.get("batch_sampler", None)
    batch_sampler = (
        None
        if batch_sampler_config is None
        else get_batch_sampler(
            dataset,
            batch_sampler_config["type"],
            sampler,
            **batch_sampler_config["params"]
        )
    )
    collate_fn = get_collate_func(data_loader_config.get("collate_fn", None))
    drop_last = data_loader_config.get("drop_last", False)
    if batch_sampler is not None:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )

    batch_size = data_loader_config["batch_size"]
    pin_memory = data_loader_config.get("pin_memory", True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def build_data_loaders(
        dataset_dict: Dict[str, Any],
        data_loader_configs: Iterable[Dict[str, Any]],
        distributed: bool = False,
):
    data_loader_list = list()
    for data_loader_config in data_loader_configs:
        dataset_id = data_loader_config.get("dataset_id", None)
        data_loader = (
            None
            if dataset_id is None or dataset_id not in dataset_dict
            else build_data_loader(
                dataset_dict[dataset_id], data_loader_config, distributed
            )
        )
        data_loader_list.append(data_loader)
    return data_loader_list


def get_dataset_dict(dataset_config):
    dataset_type = dataset_config["type"]
    dataset_dict = dict()
    if dataset_type in DATASET_DICT:
        dataset_cls_or_func = DATASET_DICT[dataset_type]
        is_torchvision = dataset_type in torchvision.datasets.__dict__
        dataset_splits_config = dataset_config["splits"]
        for split_name in dataset_splits_config.keys():
            st = time.time()
            logger.info("Loading {} data".format(split_name))
            split_config = dataset_splits_config[split_name]
            org_dataset = (
                get_torchvision_dataset(dataset_cls_or_func, split_config["params"])
                if is_torchvision
                else dataset_cls_or_func(**split_config["params"])
            )
            dataset_id = split_config["dataset_id"]
            random_split_config = split_config.get("random_split", None)
            if random_split_config is None:
                dataset_dict[dataset_id] = org_dataset
            else:
                split_dataset(
                    org_dataset, random_split_config, dataset_id, dataset_dict
                )
            logger.info("dataset_id `{}`: {} sec".format(dataset_id, time.time() - st))
    else:
        raise ValueError("dataset_type `{}` is not expected".format(dataset_type))
    return dataset_dict


def get_all_datasets(datasets_config):
    dataset_dict = dict()
    for dataset_name in datasets_config.keys():
        sub_dataset_dict = get_dataset_dict(datasets_config[dataset_name])
        dataset_dict.update(sub_dataset_dict)
    return dataset_dict

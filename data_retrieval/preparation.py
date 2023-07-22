import copy
from typing import Any, Dict, Iterable

import torch
import torchvision
from torch.utils.data import Subset, random_split
from torchvision.datasets import ImageFolder

from data_retrieval.loaders.registry import get_sample_loader
from data_retrieval.transform import TRANSFORM_CLASS_DICT
from log_infra import def_logger

logger = def_logger.getChild(__name__)


def build_transform(transform_params_config, compose_cls=None):
    if (
        not isinstance(transform_params_config, (dict, list))
        or len(transform_params_config) == 0
    ):
        return None

    if isinstance(compose_cls, str):
        compose_cls = TRANSFORM_CLASS_DICT[compose_cls]

    component_list = list()
    if isinstance(transform_params_config, dict):
        for component_key in sorted(transform_params_config.keys()):
            component_config = transform_params_config[component_key]
            params_config = component_config.get("params", dict())
            if params_config is None:
                params_config = dict()

            component = TRANSFORM_CLASS_DICT[component_config["type"]](**params_config)
            component_list.append(component)
    else:
        for component_config in transform_params_config:
            params_config = component_config.get("params", dict())
            if params_config is None:
                params_config = dict()

            component = TRANSFORM_CLASS_DICT[component_config["type"]](**params_config)
            component_list.append(component)
    return (
        torchvision.transforms.Compose(component_list)
        if compose_cls is None
        else compose_cls(component_list)
    )


def get_torchvision_dataset(dataset_cls, dataset_params_config):
    params_config = dataset_params_config.copy()
    transform_compose_cls_name = params_config.pop("transform_compose_cls", None)
    transform = build_transform(
        params_config.pop("transform_params", None),
        compose_cls=transform_compose_cls_name,
    )
    target_transform_compose_cls_name = params_config.pop(
        "target_transform_compose_cls", None
    )
    target_transform = build_transform(
        params_config.pop("target_transform_params", None),
        compose_cls=target_transform_compose_cls_name,
    )
    transforms_compose_cls_name = params_config.pop("transforms_compose_cls", None)
    transforms = build_transform(
        params_config.pop("transforms_params", None),
        compose_cls=transforms_compose_cls_name,
    )
    if "loader" in params_config:
        loader_config = params_config.pop("loader")
        loader_type = loader_config["type"]
        loader_params_config = loader_config.get("params", None)
        loader = (
            get_sample_loader(loader_type)
            if loader_params_config is None
            else get_sample_loader(loader_type, **loader_params_config)
        )
        params_config["loader"] = loader

    return dataset_cls(
        transform=transform, target_transform=target_transform, **params_config
    )


# todo: Extend this to support "label- and hierarchy-aware" splitting
def split_dataset(org_dataset, random_split_config, dataset_id, dataset_dict):
    org_dataset_length = len(org_dataset)
    logger.info(
        "Splitting `{}` dataset ({} samples in total)".format(
            dataset_id, org_dataset_length
        )
    )
    lengths = random_split_config["lengths"]
    total_length = sum(lengths)
    if total_length != org_dataset_length:
        lengths = [int((l / total_length) * org_dataset_length) for l in lengths]
        if len(lengths) > 1 and sum(lengths) != org_dataset_length:
            lengths[-1] = org_dataset_length - sum(lengths[:-1])

    manual_seed = random_split_config.get("generator_seed", None)
    sub_datasets = (
        random_split(org_dataset, lengths)
        if manual_seed is None
        else random_split(
            org_dataset, lengths, generator=torch.Generator().manual_seed(manual_seed)
        )
    )
    # Deep-copy dataset to configure transforms independently as dataset in Subset class is shallow-copied
    for sub_dataset in sub_datasets:
        sub_dataset.dataset = copy.deepcopy(sub_dataset.dataset)

    sub_splits_config = random_split_config["sub_splits"]
    assert len(sub_datasets) == len(
        sub_splits_config
    ), "len(lengths) `{}` should be equal to len(sub_splits) `{}`".format(
        len(sub_datasets), len(sub_splits_config)
    )
    for sub_dataset, sub_split_params in zip(sub_datasets, sub_splits_config):
        sub_dataset_id = sub_split_params["dataset_id"]
        logger.info(
            "new dataset_id: `{}` ({} samples)".format(sub_dataset_id, len(sub_dataset))
        )
        params_config = sub_split_params.copy()
        transform = build_transform(params_config.pop("transform_params", None))
        target_transform = build_transform(params_config.pop("transform_params", None))
        transforms = build_transform(params_config.pop("transforms_params", None))
        if hasattr(sub_dataset.train_dataset_id, "transform") and transform is not None:
            sub_dataset.train_dataset_id.transform = transform
        if (
            hasattr(sub_dataset.train_dataset_id, "target_transform")
            and target_transform is not None
        ):
            sub_dataset.train_dataset_id.target_transform = target_transform
        if hasattr(sub_dataset.train_dataset_id, "transforms") and transforms is not None:
            sub_dataset.train_dataset_id.transforms = transforms
        dataset_dict[sub_dataset_id] = sub_dataset


# TODO: Refactor asap should be the first thing we fix after the pipeline works
def get_client_subset(dataset: ImageFolder, num_clients: int, subset_id):
    logger.info(
        f"Creating subset of {dataset.__class__.__name__} for client {subset_id}"
    )
    num_classes = len(dataset.classes)
    class_indices = [[] for _ in range(num_classes)]

    for i, (_, label) in enumerate(dataset):
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
        subset = Subset(dataset, subset_indices)
        client_datasets.append(subset)
    return client_datasets[subset_id]

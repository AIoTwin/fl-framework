from collections import defaultdict
from typing import Dict, Iterable, Optional, Union

import torch
import torchvision
from torch.utils.data import Dataset

from torch.utils.model_zoo import tqdm

from data_retrieval.wrapper import BaseDatasetWrapper
from log_infra import def_logger
from torch.utils.data.sampler import BatchSampler, Sampler
from PIL import Image

logger = def_logger.getChild(__name__)
BATCH_SAMPLER_CLASS_DICT = dict()
SAMPLER_CLASS_DICT = dict()


def register_batch_sampler_class(cls):
    BATCH_SAMPLER_CLASS_DICT[cls.__name__] = cls
    return cls


@register_batch_sampler_class
class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                'sampler should be an instance of '
                'torch.utils.data.Sampler, but got sampler={}'.format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with largest number
            # of elements
            for group_id, _ in sorted(buffer_per_group.items(),
                                      key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                buffer_per_group[group_id].extend(
                    samples_per_group[group_id][:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size


class _SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _compute_aspect_ratios_slow(dataset, indices=None):
    logger.info('Your dataset doesn\'t support the fast path for '
                'computing the aspect ratios, so will iterate over '
                'the full dataset and load every image instead. '
                'This might take some time...')
    if indices is None:
        indices = range(len(dataset))

    sampler = _SubsetSampler(indices)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=sampler,
        num_workers=14,  # you might want to increase it for faster processing
        collate_fn=lambda x: x[0])
    aspect_ratios = []
    with tqdm(total=len(dataset)) as pbar:
        for _i, tuple_item in enumerate(data_loader):
            img = tuple_item[0]
            pbar.update(1)
            height, width = img.shape[-2:]
            aspect_ratio = float(width) / float(height)
            aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_custom_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        height, width = dataset.get_height_and_width(i)
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_coco_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        img_info = dataset.coco.imgs[dataset.ids[i]]
        aspect_ratio = float(img_info['width']) / float(img_info['height'])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_voc_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        # this doesn't load the data into memory, because PIL loads it lazily
        width, height = Image.open(dataset.images[i]).size
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_subset_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))

    ds_indices = [dataset.indices[i] for i in indices]
    return compute_aspect_ratios(dataset.train_dataset_id, ds_indices)


def compute_aspect_ratios(dataset, indices=None):
    target_dataset = dataset.org_dataset if isinstance(dataset, BaseDatasetWrapper) else dataset
    if hasattr(target_dataset, 'get_height_and_width'):
        return _compute_aspect_ratios_custom_dataset(target_dataset, indices)

    if isinstance(target_dataset, torchvision.datasets.CocoDetection):
        return _compute_aspect_ratios_coco_dataset(target_dataset, indices)

    if isinstance(target_dataset, torchvision.datasets.VOCDetection):
        return _compute_aspect_ratios_voc_dataset(target_dataset, indices)

    if isinstance(target_dataset, torch.utils.data.Subset):
        return _compute_aspect_ratios_subset_dataset(target_dataset, indices)

    # slow path
    return _compute_aspect_ratios_slow(target_dataset, indices)


def create_aspect_ratio_groups(dataset, k=0):
    aspect_ratios = compute_aspect_ratios(dataset)
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    groups = _quantize(aspect_ratios, bins)
    # count number of elements per group
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    logger.info('Using {} as bins for aspect ratio quantization'.format(fbins))
    logger.info('Count of instances per bin: {}'.format(counts))
    return groups


def register_sampler(cls):
    BATCH_SAMPLER_CLASS_DICT[cls.__name__] = cls
    return cls


def get_batch_sampler(dataset, class_name, *args, **kwargs):
    if class_name not in BATCH_SAMPLER_CLASS_DICT and class_name != 'BatchSampler':
        logger.info('No batch sampler called `{}` is registered.'.format(class_name))
        return None

    batch_sampler_cls = BatchSampler if class_name == 'BatchSampler' else BATCH_SAMPLER_CLASS_DICT[class_name]
    if batch_sampler_cls == GroupedBatchSampler:
        group_ids = create_aspect_ratio_groups(dataset, k=kwargs.pop('aspect_ratio_group_factor'))
        return batch_sampler_cls(*args, group_ids, **kwargs)
    return batch_sampler_cls(*args, **kwargs)


@register_sampler
class SkewedClassFederatedSampler(Sampler):
    """
        Sampler that allows you to skew classes amongs clients.
    """

    def __init__(self,
                 data_source: Dataset,
                 class_subsets: Dict,
                 rank: int):
        super(SkewedClassFederatedSampler, self).__init__(data_source)
        self.data_source = data_source

        # Ensure that the number of class subsets matches the number of workers
        if len(class_subsets) != torch.distributed.get_world_size():
            raise ValueError("Number of class subsets must equal number of workers")

        # Get the subset of classes that this worker should sample from
        self.classes_to_include = set(class_subsets[rank])

        self.indices = [i for i, sample in enumerate(data_source)
                        if sample[1] in self.classes_to_include]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


@register_sampler
class CustomIndicesSampler(Sampler):
    """
        Pass your own indices to restrict clients to a subset of the dataset.
    """
    def __init__(self, indices: Iterable[int]):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


@register_sampler
class FairClassFederatedSampler(Sampler):
    """
        Sampler that fairly assigns clients the same number of classes

        todo: Consider class sizes when assigning classes to clients
    """

    def __init__(self,
                 data_source,
                 n_classes,
                 rank):
        super(FairClassFederatedSampler, self).__init__(data_source)
        self.data_source = data_source
        self.rank = rank
        self.num_workers = torch.distributed.get_world_size()

        # Divide the classes evenly among the workers
        classes_per_worker = n_classes // self.num_workers
        self.classes_to_include = set(range(rank * classes_per_worker, (rank + 1) * classes_per_worker))

        self.indices = [i for i, sample in enumerate(data_source)
                        if sample[1] in self.classes_to_include]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


@register_sampler
class FederatedSampler(Sampler):
    """
        Distributed Sampler that (attempts to) evenly distribute samples from each class

        Restrict classes for ALL participants by passing classes_to_include
    """

    def __init__(self,
                 data_source,
                 num_replicas: int,
                 rank: int,
                 classes_to_include: Optional[Iterable[Union[str, int]]] = None):
        super(FederatedSampler, self).__init__(data_source)
        self.data_source = data_source
        self.classes_to_include = set(classes_to_include)

        self.num_replicas = num_replicas
        self.rank = rank

        if classes_to_include:
            self.indices = [i for i, sample in enumerate(data_source)
                            if sample[1] in self.classes_to_include]
        else:
            self.indices = [i for i, sample in enumerate(data_source)]

        # Divide the dataset as evenly as possible
        num_extra = len(self.indices) % self.num_replicas
        self.num_samples = len(self.indices) // self.num_replicas

        # If this sampler has one of the first `num_extra` ranks, it gets one extra sample.
        if self.rank < num_extra:
            self.num_samples += 1

        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # Generate a list of indices, one for each sample this process is responsible for.
        indices = self.indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

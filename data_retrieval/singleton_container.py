from typing import Any, Dict, Iterable, Union

from data_retrieval.retrieval import build_data_loader, get_dataset_dict
from log_infra import def_logger

logger = def_logger.getChild(__name__)


class DatasetContainer:
    """
        Emulate FL Agent for Data distribution and collecting statistics on data
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing Singleton Datset Container")
            cls._instance = super().__new__(cls)
            cls._dataset_dict = dict()
        return cls._instance

    # this allows us to extend this class with properties
    def __init__(self):
        if not hasattr(self, "dataset_dict"):
            self.dataset_dict = dict()

    @property
    def dataset_dict(self):
        return self._dataset_dict

    @dataset_dict.setter
    def dataset_dict(self, empty_dict: Dict):
        self._dataset_dict = empty_dict

    def reset(self):
        logger.debug("Resetting Dataset Conainer")
        self.dataset_dict = dict()

    def __str__(self):
        return str(self.dataset_dict)

    def __repr__(self):
        return repr(self.dataset_dict)

    def __getitem__(self, key: Union[str, int]):
        return self.dataset_dict[key]

    def __setitem__(self, key: Union[str, int], value: Any):
        self.dataset_dict[key] = value

    def get(self, key: Union[str, int]) -> Any:
        return self.dataset_dict.__getitem__(key)

    def set(self, key: Union[str, int], value: Any):
        self.dataset_dict.__setitem__(key, value)

    def update(self, d: Dict):
        self.dataset_dict.update(d)

    def keys(self):
        return self.dataset_dict.keys()

    @staticmethod
    def get_loader(dataset_id: str, loader_config: Dict[str, Any], distributed: bool = False):
        loader = build_data_loader(DatasetContainer.get(dataset_id), loader_config, distributed)
        return loader

    def _build_standard_data_container(self, datasets_config: Dict[str, Any], *args, **kwargs):
        logger.info("Building standard data container...")
        for dataset_name in datasets_config.keys():
            dataset_dict = get_dataset_dict(datasets_config[dataset_name])
            self.update(dataset_dict)

    def _build_flat_data_container(
            self, datasets_config: Dict[str, Any], client_ids: Iterable[int], *args, **kwargs
    ):
        """
        """
        logger.info("Building flat data container...")
        raise NotImplementedError

    def _build_hierarchical_data_container(self, datasets_config: Dict[str, Any], class_hierarchy_idx: Dict):
        #  future work
        raise NotImplementedError

    def get_subset(self, path: str):
        """
        """

    def build_container(self,
                        mode: str,
                        datset_config: Dict[str, Any],
                        reset: bool = True,
                        *args, **kwargs):
        if reset:
            self.reset()
        if mode == "standard":
            self._build_standard_data_container(datset_config, *args, **kwargs)
        elif mode == "flat":
            self._build_flat_data_container(datset_config, *args, **kwargs)
        else:
            raise ValueError(f"mode `{mode}` not implement")

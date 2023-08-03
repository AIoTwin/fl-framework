import concurrent.futures
import re
import time
from collections import defaultdict
from enum import Enum
from typing import Dict, Iterable, List

from spock import SpockBuilder, spock
from itertools import count
from data_retrieval import get_all_datasets
from data_retrieval.samplers.indexer import build_indices
from fl_common.threading.process_handler import SERVER_EXEC, run_script
from log_infra import def_logger, prepare_local_log_file
from misc.config_models import DatasetsConfig, LoggingConfig, ServerConfig

"""
TODO next: 
      - Format for modelling arbitrary topologies and associating node configurations with them
Note: Need top down configurations if we want central point to spawn all nodes
"""

logger = def_logger.getChild(__name__)


@spock
class HierarchicalClusteringExperimentConfig:
    """
        Extend as needed "E.g., clusters: List[ClusterConfig]"
    """
    subset_strategy: str
    base_address: str
    port_range: str
    topology: Dict[str, object]
    test_only: bool = False
    logging_config: LoggingConfig
    dataset_config: DatasetsConfig

    def __post_hook__(self):
        pattern = r'^\d{1,5}:\d{1,5}$'
        assert self.port_range and bool(re.match(pattern, self.port_range)), "port_range form: 'start:end'"


def _parse_ports(port_range: str) -> Iterable[int]:
    start, end = port_range.split(":")
    return iter(range(int(start), int(end) + 1))


def run(entry_config: HierarchicalClusteringExperimentConfig):
    prepare_local_log_file(log_file_path=entry_config.logging_config.local_logging_config.log_file_path,
                           test_only=entry_config.test_only,
                           overwrite=False)
    aggregator_futures = defaultdict(list)
    client_futures = defaultdict(list)
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        # todo: generalize topology traversal and instanciation and encapsulate
        ports = _parse_ports(entry_config.port_range)
        topology = entry_config.topology
        rank_counter = count()
        root: Dict = topology["root"]
        root_address = f"{entry_config.base_address}:{next(ports)}"
        root_config = root["base_config"]
        root_children = root["children"]
        no_children = len(root_children.get("aggregators", [])) + len(root_children.get("clients", []))
        global_agg_future = executor.submit(run_script,
                                            SERVER_EXEC,
                                            ["-c",
                                             root_config,
                                             f"--AggregatorConfig.no_children", f"{no_children}",
                                             f"--ServerConfig.server_address", f"{root_address}",
                                             f"--ServerConfig.server_type", "TorchServer"])
        aggregator_futures[f"Parent=None"] = global_agg_future
        root_children["parent_address"] = root_address
        stack: List[object] = [root_children]
        while stack:
            node: Dict = stack.pop()
            parent_address = node.get("parent_address")
            if node:
                clients = node.get("clients", [])
                aggregators = node.get("aggregators", [])
                for aggregator_node in aggregators:
                    children = aggregator_node.get("children", [])
                    base_config = aggregator_node["base_config"]
                    aggregator_address = f"{entry_config.base_address}:{next(ports)}"
                    aggregator_futures[f"Parent={parent_address}"] = executor.submit(
                        run_script,
                        SERVER_EXEC,
                        ["-c",
                         base_config,
                         f"--AggregatorConfig.no_children", f"{len(aggregator_node['children'])}",
                         f"--AggregatorConfig.parent_address", f"{parent_address}",
                         f"--ServerConfig.server_address", f"{aggregator_address}",
                         f"--ServerConfig.server_type", "BidirectionalServer",
                         ]
                    )
                    # only aggregators can have children
                    if children:
                        children["parent_address"] = aggregator_address
                        stack.append(children)
                for client_node in clients:
                    client_config = client_node["base_config"]
                    client_futures[f"Parent={parent_address}"].append(executor.submit(
                        run_script,
                        "fl_common/clients/client_exec.py",
                        ["-c",
                         client_config,
                         f"--ClientConfig.client_id", f"{next(rank_counter)}",
                         f"--ClientConfig.server_address",  f"{parent_address}"]
                    ))

    print(aggregator_futures)
    print(client_futures)


if __name__ == "__main__":
    description = "Preliminary Experiments to test the effects of client dropouts on predictive strength"
    config = SpockBuilder(HierarchicalClusteringExperimentConfig,
                          desc=description,
                          lazy=True,
                          ).generate()
    # todo: global cuda settings (e.g., cudnn.benchmark) for performance
    run(entry_config=config.HierarchicalClusteringExperimentConfig)

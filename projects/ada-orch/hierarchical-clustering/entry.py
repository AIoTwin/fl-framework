import concurrent.futures
import re
import time
from collections import defaultdict
from itertools import count
from typing import Dict, Iterable, List

from spock import SpockBuilder, spock

from fl_common.process_handler import AGGREGATOR_EXEC, run_script
from log_infra import def_logger, prepare_local_log_file
from misc.config_models import DatasetsConfig, LoggingConfig

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
        num_children = len(root_children.get("aggregators", [])) + len(root_children.get("num_clients", []))
        global_agg_future = executor.submit(run_script,
                                            AGGREGATOR_EXEC,
                                            ["-c",
                                             root_config,
                                             f"--AggregatorConfig.num_children", f"{num_children}",
                                             f"--AggregatorConfig.server_address", f"{root_address}",
                                             f"--AggregatorConfig.server_type", "TorchServerWithCentralizedEval"])
        aggregator_futures[f"Parent=None"] = global_agg_future
        root_children["parent_address"] = root_address
        time.sleep(5)
        stack: List[object] = [root_children]
        while stack:
            node: Dict = stack.pop()
            parent_address = node.get("parent_address")
            if node:
                aggregators = node.get("aggregators", [])
                for aggregator_node in aggregators:
                    children = aggregator_node.get("children", [])
                    base_config = aggregator_node["base_config"]
                    aggregator_address = f"{entry_config.base_address}:{next(ports)}"
                    num_children = len(children.get("aggregators", [])) + children.get("num_clients", [])
                    aggregator_futures[f"Parent={parent_address}"] = executor.submit(
                        run_script,
                        AGGREGATOR_EXEC,
                        ["-c",
                         base_config,
                         f"--AggregatorConfig.num_children", f"{num_children}",
                         f"--AggregatorConfig.parent_address", f"{parent_address}",
                         f"--AggregatorConfig.server_address", f"{aggregator_address}",
                         f"--AggregatorConfig.server_type", "TorchServer",
                         "--StrategyConfig.strategy_params",
                         "{'min_available_clients': "
                         + f"{num_children},"
                         + "'min_fit_clients': "
                         + f"{num_children},"
                         + "'min_evaluate_clients': "
                         + f"{num_children}"
                         + "}",
                         ]
                    )
                    time.sleep(5)
                    # only aggregators can have children
                    if children:
                        children["parent_address"] = aggregator_address
                        for client_node_id in range(children.get("num_clients", [])):
                            client_config = children["base_config"]
                            failures_at_round = children["failures_at_round"]
                            client_id = next(rank_counter)
                            if client_id < len(failures_at_round):
                                client_futures[f"Parent={aggregator_address}"].append(executor.submit(
                                    run_script,
                                    "fl_common/clients/client_exec.py",
                                    ["-c",
                                     client_config,
                                     f"--ClientConfig.client_id", f"h{client_id}",
                                     f"--ClientConfig.server_address", f"{aggregator_address}",
                                     "--ClientConfig.client_type", "UnreliableClient",
                                     "--ClientConfig.client_params",
                                     "{'fail_at_round': "
                                     + f"{failures_at_round[client_id]}"
                                     + "}",
                                     ]
                                ))
                            else:
                                client_futures[f"Parent={aggregator_address}"].append(executor.submit(
                                    run_script,
                                    "fl_common/clients/client_exec.py",
                                    ["-c",
                                     client_config,
                                     f"--ClientConfig.client_id", f"h{client_id}",
                                     f"--ClientConfig.server_address", f"{aggregator_address}",
                                     "--ClientConfig.client_type", "TorchClient"]
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

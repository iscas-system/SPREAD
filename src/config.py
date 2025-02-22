import json
from log import info
import os
import sys
from itertools import count
from typing import Tuple, Dict, List, Optional, Set, DefaultDict
from collections import defaultdict
from object import ModelName, model_name_strs, SchedulerEnum, GPUType, SimulatingMethod, NodeType, Node, GPU

job_deploy_specs = (False, 1), (False, 2), (True, 2), (False, 4), (True, 4) # cross_node, worker_count

class Config:
    def __init__(self, config_path: str):
        with open(config_path) as c:
            d = json.load(c)
        self.license_path: Optional[str] = d.get("license_path", None)
        self.using_default_license: bool = d["use_default_license"]
        self.__init_license()
        self.session_id: str = d["session_id"]
        self.enabled_scheduler_names: List[str] = d["enabled_scheduler_names"]
        self.simulating_method: SimulatingMethod = SimulatingMethod(d.get("simulating_method", "Trace"))
        self.simulating_method_config: Dict = d.get("simulating_method_config", dict())
        self.multiprocessing: bool = d["multiprocessing"]
        self.log_to_stdout: bool = d["log_to_stdout"]

        self.data_source_configs: Dict[str, 'DataSourceConfig'] = dict()
        for cn, c in d["data_source_configs"].items():
            self.data_source_configs[cn] = DataSourceConfig(
                name=cn,
                submit_table_path=c.get("submit_table_path", "./data/pai_job_submit_table.csv"),
                data_range=c["data_range"],
                init_job_data_seed=c["init_job_data_seed"],
                job_count=c["job_count"],
                submit_at_beginning=c["submit_at_beginning"],
                submit_scale_factor=c["submit_scale_factor"],
                filter_replicas=c.get("filter_replicas", [1]),
                comp_distribution=c.get("comp_distribution", "ali"),
                mono_job_data_path=c["mono_job_data_path"]
            )
        self.enabled_data_source_configs: List[str] = d["enabled_data_source_configs"]
        self.model_configs: Dict[ModelName, ModelConfig] = dict()
        for model_name_str, model_config_dict in d["models"].items():
            if model_name_str not in model_name_strs:
                continue
            self.model_configs[ModelName[model_name_str]] = ModelConfig(
                model_name=ModelName[model_name_str],
                batch_sizes=model_config_dict["batch_sizes"],
                preemptive_overhead=model_config_dict["preemptive_overhead"]
            )
        self.cluster_configs: Dict[str, 'ClusterConfig'] = dict()
        for cn, c in d["cluster_configs"].items():
            self.cluster_configs[cn] = ClusterConfig.from_node_specs(
                name=cn,
                node_types=[NodeType(node_type_name=node_type_name, GPUs={GPUType(g): cnt for g, cnt in GPUs.items()}) for node_type_name, GPUs in c["node_type"].items()],
                node_type_to_count={node_type_name: node_count for node_type_name, node_count in c["nodes"].items()}
            )
        self.enabled_cluster_configs: List[str] = d["enabled_cluster_configs"]
        self.default_scheduling_preemptive_interval: int = d["default_scheduling_preemptive_interval"]
        self.schedulers: Dict[str, SchedulerDescription] = dict()
        for sc in d["schedulers"]:
            self.schedulers[sc["name"]] = SchedulerDescription(
                name=sc["name"],
                scheduler_enum=SchedulerEnum[sc["scheduler_enum"]],
                config=sc["config"]
            )

    def __init_license(self):
        if self.using_default_license:
            return
        GRB_LICENSE_FILE_ENV_KEY = "GRB_LICENSE_FILE"
        if self.license_path is not None:
            if not os.path.exists(self.license_path):
                info(f"license_location: {self.license_path} not exists.")
                sys.exit(-1)
            os.environ[GRB_LICENSE_FILE_ENV_KEY] = self.license_path
            info(f"using specified license file in {self.license_path}")
        else:
            info(f"using license file specified in environment variable {os.environ[GRB_LICENSE_FILE_ENV_KEY]}")


class SchedulerDescription:
    def __init__(self, name: str, scheduler_enum: SchedulerEnum, config: Dict):
        self.name: str = name
        self.scheduler_enum: SchedulerEnum = scheduler_enum
        self.config: Dict = config


class ModelConfig:
    def __init__(self, model_name: ModelName, batch_sizes: List[int], preemptive_overhead: Tuple[int, int]):
        self.model_name: ModelName = model_name
        self.batch_sizes: List[int] = batch_sizes
        self.preemptive_overhead: Tuple[int, int] = preemptive_overhead


def loaded_config(get_config_func):
    config_object: Optional[Config] = None

    def get_config_wrapped(config_path: Optional[str] = None):
        nonlocal config_object
        if config_object is not None and config_path is None:
            return config_object
        config_object = get_config_func(config_path)
        return config_object

    return get_config_wrapped


@loaded_config
def get_config(config_path: Optional[str] = None):
    return Config(config_path=config_path)


class ClusterConfig:

    @classmethod
    def from_node_specs(cls, name: str, node_types: List[NodeType], node_type_to_count: Dict[str, int]) -> 'ClusterConfig':
        node_types_dict = {node_type.node_type_name: node_type for node_type in node_types}
        nodes = []
        counter = count(0)
        for node_type_name, c in node_type_to_count.items():
            nodes.extend([Node(node_type_name, next(counter)) for _ in range(c)])
        GPU_types: Set[GPUType] = set()
        for nt_name, nt in node_types_dict.items():
            GPU_types.update(nt.GPU_types)

        GPUs = defaultdict(list)
        GPU_ID_to_node_id = dict()
        GPU_idx_counter = count(0)
        for node in nodes:
            node_type = node_types_dict[node.node_type_name]
            for GPU_type, c in node_type.GPUs.items():
                for i in range(c):
                    GPU_type = GPUType(GPU_type)
                    g = GPU.from_idx_node_id(GPU_type=GPU_type, idx=next(GPU_idx_counter), node_id=node.node_id)
                    GPUs[GPU_type].append(g)
                    GPU_ID_to_node_id[g.GPU_ID] = node.node_id
        return cls(name=name, GPUs=GPUs, GPU_ID_to_node_id=GPU_ID_to_node_id)

    @classmethod
    def from_GPU_specs(cls, name: str, GPU_ID_type_node_id: Dict[str, Tuple[GPUType, str]]) -> 'ClusterConfig':
        GPUs = defaultdict(list)
        GPU_ID_to_node_id = dict()
        for GPU_ID, spec in GPU_ID_type_node_id.items():
            GPU_type, node_id = spec
            g = GPU(GPU_type=GPU_type, GPU_ID=GPU_ID, node_id=node_id)
            GPUs[GPU_type].append(g)
            GPU_ID_to_node_id[GPU_ID] = node_id
        return cls(name=name, GPUs=GPUs, GPU_ID_to_node_id=GPU_ID_to_node_id)

    def __init__(self, name: str, GPUs: DefaultDict[GPUType, List[GPU]], GPU_ID_to_node_id: Dict[str, str]):
        self.name: str = name
        self.GPU_types: Set[GPUType] = set(GPUs.keys())

        self.GPUs: DefaultDict[GPUType, List[GPU]] = GPUs
        self.GPU_ID_to_node_id: Dict[str, str] = GPU_ID_to_node_id
        self.GPU_IDs: List[str] = list()
        self.GPU_ID_to_GPU: Dict[str, GPU] = dict()
        for _, typed_GPUs in GPUs.items():
            for g in typed_GPUs:
                self.GPU_IDs.append(g.GPU_ID)
                self.GPU_ID_to_GPU[g.GPU_ID] = g
        self.GPU_IDs.sort()

    def get_GPU(self, GPU_ID) -> GPU:
        return self.GPU_ID_to_GPU[GPU_ID]

class DataSourceConfig:
    def __init__(self,
                 name: str,
                 submit_table_path: str,
                 data_range: Tuple[int, int],
                 init_job_data_seed: int,
                 job_count: int,
                 submit_at_beginning: bool,
                 submit_scale_factor: Optional[float],
                 filter_replicas: List[int],
                 comp_distribution: str,
                 mono_job_data_path: str
                 ):
        self.name: str = name
        self.submit_table_path: str = submit_table_path
        self.data_range: Tuple[int, int] = data_range
        self.init_job_data_seed: int = init_job_data_seed
        self.job_count: int = job_count
        self.submit_at_beginning: bool = submit_at_beginning
        self.submit_scale_factor: float = submit_scale_factor
        self.filter_replicas: List = filter_replicas
        self.comp_distribution: str = comp_distribution
        self.mono_job_data_path: str = mono_job_data_path

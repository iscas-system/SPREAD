import json
from log import info
import os
import sys
from typing import Tuple, Dict, List, Optional, Set

from object import ModelName, model_name_strs, SchedulerEnum, GPUType, SimulatingMethod


class Config:
    """
        {
            "license_path": null,
            "cluster_configs": {
                "cluster_1": {
                    "GPUs": {
                        "RTX_2080Ti": 4,
                        "Tesla_T4": 3
                    }
                },
                "cluster_2": {
                    "GPUs": {
                        "RTX_2080Ti": 4,
                        "Tesla_T4": 3
                    },
                    "enabled_scheduler_names": ["MMKP", "RoundRobin", "BinPacking", "Tiresias", "Themis", "Optimus"]
                }
            },
            "enabled_cluster_configs": ["cluster_1", "cluster_2"],
            "models": {
                "BertBase": {
                    "batch_sizes": [8, 16, 32, 64, 128],
                    "preemptive_overhead": [10, 15]
                },
                "LSTM": {
                    "batch_sizes": [10, 20, 40, 80],
                    "preemptive_overhead": [5, 10]
                },
                "ResNet50": {
                    "batch_sizes": [8, 16, 32, 64],
                    "preemptive_overhead": [20, 30]
                },
                "ResNet18": {
                    "batch_sizes": [8, 16, 32, 64],
                    "preemptive_overhead": [10, 15]
                },
                "YoloV5S": {
                    "batch_sizes": [8, 16, 32, 64, 128],
                    "preemptive_overhead": [20, 25]
                },
                "MobileNet": {
                    "batch_sizes": [8, 16, 32, 64, 128],
                    "preemptive_overhead": [5, 15]
                }
            },
            "data_source_configs": {
                "data_source_1": {
                    "data_range": [100, 500],
                    "job_count": 300,
                    "submit_at_beginning": false,
                    "enabled_scheduler_names": ["MMKP", "RoundRobin", "BinPacking"]
                },
                "data_source_2": {
                    "data_range": [100, 500],
                    "job_count": 300,
                    "submit_at_beginning": false,
                    "enabled_scheduler_names": ["MMKP", "Tiresias", "Themis", "Optimus"]
                }
            },
            "enabled_data_source_configs": ["data_source_1", "data_source_2"],
            "default_scheduling_interval": 360，
            "schedulers": [
                {
                    "name": "MMKP_SRSF",
                    "scheduler_enum": "MMKP",
                    "config": {
                        "priority_type": "SRSF"
                    }
                },
                {
                    "name": "MMKP_FCFS",
                    "scheduler_enum": "MMKP",
                    "config": {
                        "priority_type": "FCFS"
                    }
                },
                {
                    "name": "RoundRobin_SRSF",
                    "scheduler_enum": "RoundRobin",
                    "config": {
                        "priority_type": "SRSF"
                    }
                },
                {
                    "name": "RoundRobin_FCFS",
                    "scheduler_enum": "RoundRobin",
                    "config": {
                        "priority_type": "FCFS"
                    }
                },
                {
                    "name": "BinPacking_1",
                    "scheduler_enum": "BinPacking",
                    "config": {}
                },
                {
                    "name": "Optimus_1",
                    "scheduler_enum": "Optimus",
                    "config": {}
                },
                {
                    "name": "Tiresias_1",
                    "scheduler_enum": "Tiresias",
                    "config": {}
                }
            ],
            "enabled_scheduler_names": ["RoundRobin_FCFS"]
        }
    """

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

        self.data_source_configs: Dict[str, 'DataSourceConfig'] = dict()
        for cn, c in d["data_source_configs"].items():
            self.data_source_configs[cn] = DataSourceConfig(
                name=cn,
                submit_table_path=c.get("submit_table_path", "./data/pai_job_submit_table.csv"),
                data_range=c["data_range"],
                init_job_data_seed=c["init_job_data_seed"],
                job_count=c["job_count"],
                submit_at_beginning=c["submit_at_beginning"],
                filter_replicas=c.get("filter_replicas", [1]),
                enabled_scheduler_names=c.get("enabled_scheduler_names", self.enabled_scheduler_names),
                enable_plot=c.get("enable_plot", False),
                comp_distribution=c.get("comp_distribution", "ali")
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
            self.cluster_configs[cn] = ClusterConfig(
                name=cn,
                GPUs={GPUType[GPU_type_str]: count for GPU_type_str, count in c["GPUs"].items()},
                enabled_scheduler_names=c.get("enabled_scheduler_names", self.enabled_scheduler_names),
                enable_plot=c.get("enable_plot", False)
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
        self.enabled_scheduler_names: List[str] = d["enabled_scheduler_names"]

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
    def __init__(self, name: str, GPUs: Dict[GPUType, int], enabled_scheduler_names: List[str], enable_plot: bool):
        self.name: str = name
        self.GPUs: Dict[GPUType, int] = GPUs
        self.GPU_types: Set[GPUType] = set(GPUs.keys())
        self.enabled_scheduler_names: List[str] = enabled_scheduler_names
        self.enable_plot: bool = enable_plot


class DataSourceConfig:
    def __init__(self,
                 name: str,
                 submit_table_path: str,
                 data_range: Tuple[int, int],
                 init_job_data_seed: int,
                 job_count: int,
                 submit_at_beginning: bool,
                 filter_replicas: List[int],
                 enabled_scheduler_names: List[str],
                 enable_plot: bool,
                 comp_distribution: str,
                 ):
        self.name: str = name
        self.submit_table_path: str = submit_table_path
        self.data_range: Tuple[int, int] = data_range
        self.init_job_data_seed: int = init_job_data_seed
        self.job_count: int = job_count
        self.submit_at_beginning: bool = submit_at_beginning
        self.filter_replicas: List = filter_replicas
        self.enabled_scheduler_names: List[str] = enabled_scheduler_names
        self.enable_plot: bool = enable_plot
        self.comp_distribution: str = comp_distribution

import json
import logging
import os
import sys
from typing import Tuple, Dict, List, Optional

from object import ModelName, SchedulerEnum, GPUType


class Config:
    """
        {
            "license_path": null,
            "cluster_config": {
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
                    "enabled_schedulers": ["MMKP", "RoundRobin", "BinPacking", "Tiresias", "Themis", "Optimus"]
                }
            },
            "enabled_cluster_config": ["cluster_1", "cluster_2"],
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
                    "use_all_computation": false,
                    "enabled_schedulers": ["MMKP", "RoundRobin", "BinPacking"]
                },
                "data_source_2": {
                    "data_range": [100, 500],
                    "job_count": 300,
                    "submit_at_beginning": false,
                    "use_all_computation": true,
                    "enabled_schedulers": ["MMKP", "Tiresias", "Themis", "Optimus"]
                }
            },
            "enabled_data_source_config": ["data_source_1", "data_source_2"],
            "scheduling_interval": 360，
            "schedulers": [
                {
                    "name": "MMKP_1",
                    "scheduler_enum": "MMKP",
                    "config": {}
                },
                {
                    "name": "RoundRobin_1",
                    "scheduler_enum": "RoundRobin",
                    "config": {}
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
            ]
        }
    """

    def __init__(self, config_path: str):
        with open(config_path) as c:
            d = json.load(c)
        self.license_path: Optional[str] = d.get("license_path", None)

        self.data_source_configs: Dict[str, 'DataSourceConfig'] = dict()
        for cn, c in d["data_source_configs"].items():
            self.data_source_configs[cn] = DataSourceConfig(
                name=cn,
                submit_table_path=c.get("submit_table_path", "./data/pai_job_submit_table.csv"),
                data_range=c["data_range"],
                job_count=c["job_count"],
                submit_at_beginning=c["submit_at_beginning"],
                filter_replicas=c.get("filter_replicas", [1]),
                enabled_schedulers=c["enabled_schedulers"]
            )
        self.enabled_data_source_configs: List[str] = d["enabled_data_source_configs"]
        self.model_configs: Dict[ModelName, ModelConfig] = dict()
        for model_name_str, model_config_dict in d["models"]:
            self.model_configs[ModelName(model_name_str)] = ModelConfig(
                model_name=ModelName(model_name_str),
                batch_sizes=model_config_dict["batch_sizes"],
                preemptive_overhead=model_config_dict["preemptive_overheads"]
            )
        self.cluster_configs: Dict[str, 'ClusterConfig'] = dict()
        for cn, c in d["cluster_configs"].items():
            self.cluster_configs[cn] = ClusterConfig(
                name=cn,
                GPUs={GPUType(GPU_type_str): count for GPU_type_str, count in c["GPUs"].items()},
                enabled_schedulers=[SchedulerEnum(s) for s in d.get("enabled_schedulers", [])]
            )
        self.schedulers: List[SchedulerDescription] = list()
        for sn, sc in d["schedulers"].items():
            self.schedulers.append(SchedulerDescription(
                name=sc["name"],
                scheduler_enum=sc["scheduler_enum"],
                config=sc["config"]
            ))

    def __init_license(self):
        GRB_LICENSE_FILE_ENV_KEY = "GRB_LICENSE_FILE"
        if self.license_path is not None:
            if not os.path.exists(self.license_path):
                logging.fatal(f"license_location: {self.license_path} not exists.")
                sys.exit(-1)
            os.environ[GRB_LICENSE_FILE_ENV_KEY] = self.license_path
            logging.info(f"using specified license file in {self.license_path}")
        else:
            logging.info(f"using license file specified in environment variable {os.environ[GRB_LICENSE_FILE_ENV_KEY]}")


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
    def __init__(self, name: str, GPUs: Dict[GPUType, int], enabled_schedulers: List[SchedulerEnum]):
        self.name: str = name
        self.GPUs: Dict[GPUType, int] = GPUs
        self.enabled_schedulers: List[SchedulerEnum] = enabled_schedulers


class DataSourceConfig:
    def __init__(self,
                 name: str,
                 submit_table_path: str,
                 data_range: Tuple[int, int],
                 job_count: int,
                 submit_at_beginning: bool,
                 filter_replicas: List[int],
                 enabled_schedulers: List[SchedulerEnum]
                 ):
        self.name: str = name
        self.submit_table_path: str = submit_table_path
        self.data_range: Tuple[int, int] = data_range
        self.job_count: int = job_count
        self.submit_at_beginning: bool = submit_at_beginning
        self.filter_replicas: List = filter_replicas
        self.enabled_schedulers: List[SchedulerEnum] = enabled_schedulers

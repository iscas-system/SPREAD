import json
from collections import namedtuple
from enum import Enum
from typing import List, Optional

from common import *
from config import get_config
from data_source import DataSource, MonoJobExecInfoLoader, MonoJobExecInfo, TrainOrInference
from object import GPUType, ModelName

GPUSpec = namedtuple(typename="AccSpec", field_names=["name", "memory", "line", "marker", "label"])
GBi = 1024 * 1024 * 1024  # 1024 * 1024 * 1024 B
StableWarmupStartRatio = 3
NodeCPUCapacity = 40


def get_GPU_spec(gpu_type: GPUType) -> GPUSpec:
    RTX_2080Ti_spec = GPUSpec(name="RTX 2080Ti", memory=11 * GBi, line="-", marker="o", label="RTX 2080Ti")
    Tesla_T4_spec = GPUSpec(name="Tesla T4", memory=15 * GBi, line="--", marker="^", label="Tesla T4")
    return {
        GPUType.RTX_2080Ti: RTX_2080Ti_spec,
        GPUType.Tesla_T4: Tesla_T4_spec
    }[gpu_type]


def model_name_spec(model_name: ModelName):
    return {
        ModelName.ResNet50: {
            "label": "ResNet50"
        },
        ModelName.ResNet18: {
            "label": "ResNet18"
        },
        ModelName.LSTM: {
            "label": "LSTM"
        },
        ModelName.BertBase: {
            "label": "BertBase"
        },
        ModelName.MobileNet: {
            "label": "MobileNetV2"
        },
        ModelName.InceptionV3: {
            "label": "InceptionV3"
        },
        ModelName.YoloV5S: {
            "label": "YoloV5"
        },
        ModelName.EfficientNet: {
            "label": "EfficientNet"
        }
    }[model_name]


def batch_size_color(batch_size: int) -> str:
    return {
        4: colors[6],
        5: colors[6],
        8: colors[7],
        10: colors[7],
        16: colors[0],
        20: colors[0],
        32: colors[1],
        40: colors[1],
        64: colors[2],
        80: colors[2],
        128: colors[3],
        160: colors[3],
    }[batch_size]


def batch_size_idx_color(batch_size_idx: int) -> str:
    return {
        0: colors[0],
        1: colors[1],
        2: colors[2],
        3: colors[3],
    }[batch_size_idx]


def batch_size_idx_label(batch_size_idx: int) -> str:
    return {
        0: r"bs. 1$\times$",
        1: r"bs. 2$\times$",
        2: r"bs. 3$\times$",
        3: r"bs. 4$\times$",
    }[batch_size_idx]


def batch_size_level_color(batch_size_level: int) -> str:
    return colors[batch_size_level]


class JobExecInfo:
    def __init__(self,
                 model_name: ModelName,
                 gpu_type: GPUType,
                 worker_count: int,
                 cross_node: bool,
                 train_or_inference: TrainOrInference,
                 batch_size: int,
                 total_batch_size: int,
                 cpu_batch_size: int,
                 acc_batch_size: int,
                 computation_proportion: int,
                 cpu_limit: Optional[int],
                 time_str: str,
                 raw_json: dict,
                 cpu_worker_count: Optional[int] = None,
                 gpu_worker_count: Optional[int] = None,
                 ):
        self.model_name: ModelName = model_name
        self.gpu_type: GPUType = gpu_type
        self.worker_count: int = worker_count
        self.cross_node: bool = cross_node
        self.train_or_inference: TrainOrInference = train_or_inference
        self.batch_size: int = batch_size
        self.total_batch_size: int = total_batch_size
        self.cpu_batch_size: int = cpu_batch_size
        self.acc_batch_size: int = acc_batch_size
        self.computation_proportion: int = computation_proportion
        self.cpu_limit: Optional[int] = cpu_limit
        self.time_str: str = time_str
        self.raw_json: dict = raw_json
        self.cpu_worker_count: Optional[int] = cpu_worker_count
        self.gpu_worker_count: Optional[int] = gpu_worker_count
        self.__parse_raw_json()

    def __parse_raw_json(self):
        self.iteration_count: int = self.raw_json["iteration_count"]
        self.iteration_intervals: List[int] = self.raw_json["iteration_intervals"]
        self.total_time_ns: int = self.raw_json["total_time_ns"]
        self.mem_infos: List[List[int]] = self.raw_json["mem_infos"]
        self.utilization: List[int] = self.raw_json["utilization"]
        self.device_type: str = self.raw_json.get("device_type", "gpu")
        memories = [mem_info[-1] - mem_info[0] for mem_info in self.mem_infos]
        self.max_memory_consumption: int = 0 if len(memories) == 0 else max(memories)
        self.most_memory_consumption: int = 0 if len(memories) == 0 else most(memories)
        self.stabled_iteration_intervals: List[int] = self.iteration_intervals[
                                                      len(self.iteration_intervals) // StableWarmupStartRatio:]
        mean_iteration_intervals = np.mean(self.stabled_iteration_intervals)
        self.stabled_iteration_intervals = list(
            filter(lambda iteration_interval: iteration_interval < 50 * mean_iteration_intervals,
                   self.stabled_iteration_intervals))
        self.avg_stabled_iteration_interval: int = int(np.mean(self.stabled_iteration_intervals))
        self.stabled_utilization: List[int] = self.utilization[len(self.utilization) // StableWarmupStartRatio:]
        self.avg_stabled_utilization: float = float(np.mean(self.stabled_utilization))

    def to_dict(self):
        return {
            "model_name": self.model_name.name,
            "gpu_type": self.gpu_type.value.name,
            "worker_count": self.worker_count,
            "cross_node": self.cross_node,
            "train_or_inference": self.train_or_inference.name,
            "batch_size": self.batch_size,
            "total_batch_size": self.total_batch_size,
            "cpu_batch_size": self.cpu_batch_size,
            "acc_batch_size": self.acc_batch_size,
            "computation_proportion": self.computation_proportion,
            "cpu_limit": self.cpu_limit,
            "time_str": self.time_str,
            "cpu_worker_count": self.cpu_worker_count,
            "gpu_worker_count": self.gpu_worker_count,
            "iteration_count": self.iteration_count,
            "device_type": self.device_type,
            "max_memory_consumption": self.max_memory_consumption,
            "avg_stabled_iteration_interval": self.avg_stabled_iteration_interval,
            "avg_stabled_utilization": self.avg_stabled_utilization
        }


def do_test():
    c = get_config("../configs/MMKP_config.json")
    d = DataSource(data_source_config=c.data_source_configs["data_source_ali_fix_new"],
                   enabled_GPU_types={GPUType.RTX_2080Ti})
    print(d.job_specs[0].total_iterations)
    e = DataSource(data_source_config=c.data_source_configs["data_source_phi_uni"],
                   enabled_GPU_types={GPUType.RTX_2080Ti})
    print(e.job_specs[0].total_iterations)
    infos = MonoJobExecInfoLoader.extract(e.mono_job_data[ModelName.MEALV2], train_or_inference=TrainOrInference.train,
                                          batch_size=128, worker_count=4, cross_node=True)
    print(infos)


if __name__ == '__main__':
    do_test()

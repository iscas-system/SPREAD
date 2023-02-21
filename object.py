import re
from enum import Enum
from typing import Optional, Dict


GBi = 1024 * 1024 * 1024  # 1024 * 1024 * 1024 B

CompCapacity = 20
MemoryUnit = GBi // 2

def to_normalized_memory(real_memory):
    if real_memory % MemoryUnit == 0:
        return real_memory // MemoryUnit
    return (real_memory // MemoryUnit) + 1

def to_real_comp(normalized_comp: int):
    return normalized_comp * (100 // CompCapacity)


def to_normalized_comp(real_comp: int):
    return real_comp // (100 // CompCapacity)


class GPUType(Enum):
    Tesla_T4 = "Tesla_T4"
    RTX_2080Ti = "RTX_2080Ti"

    @staticmethod
    def label(GPU_type: 'GPUType'):
        return {
            GPUType.RTX_2080Ti: "RTX 2080Ti",
            GPUType.Tesla_T4: "Tesla T4",
        }[GPU_type]

    @staticmethod
    def comp_power_label(GPU_type: 'GPUType'):
        return {
            GPUType.RTX_2080Ti: "13.45\nTFLOPS",
            GPUType.Tesla_T4: "8.14\nTFLOPS",
        }[GPU_type]

    @staticmethod
    def mem_label(GPU_type: 'GPUType'):
        return {
            GPUType.RTX_2080Ti: "11 GB",
            GPUType.Tesla_T4: "15 GB",
        }[GPU_type]

    @staticmethod
    def real_memory(GPU_type: 'GPUType'):
        return {
            GPUType.RTX_2080Ti: 11 * GBi,
            GPUType.Tesla_T4: 15 * GBi,
        }[GPU_type]

    @staticmethod
    def normalized_memory(GPU_type: 'GPUType'):
        return to_normalized_memory(GPUType.real_memory(GPU_type))

    @staticmethod
    def line(GPU_type: 'GPUType'):
        return {
            GPUType.RTX_2080Ti: '-',
            GPUType.Tesla_T4: '--',
        }[GPU_type]

    @staticmethod
    def marker(GPU_type: 'GPUType'):
        return {
            GPUType.RTX_2080Ti: 'o',
            GPUType.Tesla_T4: '^',
        }[GPU_type]

    @staticmethod
    def service_factor(GPU_type: 'GPUType'):
        return {
            GPUType.RTX_2080Ti: 2,
            GPUType.Tesla_T4: 1,
        }[GPU_type]


class ModelName(Enum):
    ResNet50 = "ResNet50"
    ResNet18 = "ResNet18"
    LSTM = "LSTM"
    BertBase = "BertBase"
    MobileNet = "MobileNet"
    # VGG16 = "VGG16"
    InceptionV3 = "InceptionV3"
    YoloV5S = "YoloV5S"
    EfficientNet = "EfficientNet"
    MobileNetV2 = "MobileNetV2"
    SqueezeNet = "SqueezeNet"
    ShuffleNet = "ShuffleNet"
    HarDNet = "HarDNet"
    GhostNet = "GhostNet"
    MEALV2 = "MEALV2"



model_name_strs = [model_name.name for model_name in ModelName]


class JobSpec:
    def __init__(self,
                 job_ID: str,
                 model_name: ModelName,
                 batch_size: int,
                 submit_time_nano: int,
                 plan_GPU: int,
                 run_time_nano: int,
                 total_iterations: float,
                 worker_count: int,
                 ):
        self.job_ID: str = job_ID
        self.model_name: ModelName = model_name
        self.batch_size: int = batch_size
        self.submit_time_nano: int = submit_time_nano
        self.plan_GPU: int = plan_GPU
        self.plan_worker_count: int = worker_count
        self.plan_comp = (plan_GPU // (100//CompCapacity)) // self.plan_worker_count
        self.run_time_nano: int = run_time_nano
        self.total_iterations: float = total_iterations

    def to_dict(self) -> Dict:
        return {
            "job_ID": self.job_ID,
            "model_name": self.model_name.name,
            "batch_size": int(self.batch_size),
            "submit_time": self.submit_time_nano / 1e9,
            "plan_GPU": int(self.plan_GPU),
            "plan_worker_count": self.plan_worker_count,
            "plan_comp": int(self.plan_comp),
            "run_time": self.run_time_nano / 1e9,
            "total_iterations": self.total_iterations
        }


class Job:
    def __init__(self, job_ID: str, submit_time: int, remaining_iterations: float, start_time: Optional[int]=None, completion_time: Optional[int]=None):
        self.job_ID: str = job_ID
        self.remaining_iterations: float = remaining_iterations
        self.submit_time: int = submit_time
        self.start_time: Optional[int] = start_time
        self.completion_time: Optional[int] = completion_time

    def __hash__(self):
        return hash(self.job_ID)

    def __eq__(self, other):
        return self.job_ID == other.job_ID

    def __ne__(self, other):
        return self.job_ID != other.job_ID


class Task:
    def __init__(self, job_ID: str, task_idx: int):
        self.job_ID: str = job_ID
        self.task_idx: int = task_idx
        self.task_ID = f"{self.job_ID}|task_{self.task_idx}"

    @staticmethod
    def task_ID_to_job_ID(task_ID: str) -> str:
        return task_ID.split("|")[0]

    @staticmethod
    def from_task_ID(task_ID: str) -> 'Task':
        groups = re.search(r"(.*)\|task_(\d+)", task_ID)
        assert groups is not None
        job_ID = str(groups.group(1))
        task_idx = int(groups.group(2))
        return Task(job_ID=job_ID, task_idx=task_idx)

    def __hash__(self):
        return hash(self.task_ID)

    def __eq__(self, other):
        return self.task_ID == other.task_ID

    def __ne__(self, other):
        return self.task_ID != other.task_ID


class GPU:
    def __init__(self, GPU_type: GPUType, GPU_ID: str, node_id: str):
        self.GPU_type: GPUType = GPU_type
        self.GPU_ID: str = GPU_ID
        self.node_id: str = node_id

    @staticmethod
    def from_idx_node_id(GPU_type: GPUType, idx: int, node_id: str):
        GPU_ID: str = f"{GPU_type.name}_{idx}"
        return GPU(GPU_type=GPU_type, GPU_ID=GPU_ID, node_id=node_id)

    def __str__(self):
        return f"GPU[ID={self.idx}, node_id={self.node_id}]"

class NodeType:
    def __init__(self, node_type_name: str, GPUs: Dict[GPUType, int]):
        self.node_type_name: str = node_type_name
        self.GPUs: Dict[GPUType, int] = GPUs
        self.GPU_types = set(GPUs.keys())

class Node:
    def __init__(self, node_type_name: str, idx: int):
        self.node_type_name: str = node_type_name
        self.idx: int = idx
        self.node_id: str = f"{self.node_type_name}_{self.idx}"

    def __str__(self):
        return f"Node[node_id={self.node_id}]"

class SchedulerEnum(Enum):
    MMKP = "MMKP"
    RoundRobin = "RoundRobin"
    BinPacking = "BinPacking"
    Tiresias = "Tiresias"
    Themis = "Themis"
    Optimus = "Optimus"
    KubeShare = "KubeShare"
    BestFit = "BestFit"
    Gavel = "Gavel"
    Kubernetes = "Kubernetes"


class SimulatingMethod(Enum):
    Trace = "Trace"
    RandomPlacement = "RandomPlacement"
    RandomPlacementSelector = "RandomPlace"

class SolverEnum(Enum):
    MMKP = "MMKP"


class ProfitEnum(Enum):
    ComprehensiveUtilization = "ComprehensiveUtilization"


class PriorityType(Enum):
    SRSF = "SRSF"
    FCFS = "FCFS"

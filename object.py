from enum import Enum
from typing import Optional

GBi = 1024 * 1024 * 1024  # 1024 * 1024 * 1024 B

CompCapacity = 20
MemoryUnit = GBi


class GPUType(Enum):
    Tesla_T4 = "Tesla_T4"
    RTX2080_Ti = "RTX2080_Ti"

    @staticmethod
    def label(GPU_type: 'GPUType'):
        return {
            GPUType.RTX2080_Ti: "RTX 2080Ti",
            GPUType.Tesla_T4: "Tesla T4",
        }[GPU_type]

    @staticmethod
    def comp_power_label(GPU_type: 'GPUType'):
        return {
            GPUType.RTX2080_Ti: "13.45\nTFLOPS",
            GPUType.Tesla_T4: "8.14\nTFLOPS",
        }[GPU_type]

    @staticmethod
    def mem_label(GPU_type: 'GPUType'):
        return {
            GPUType.RTX2080_Ti: "11 GB",
            GPUType.Tesla_T4: "15 GB",
        }[GPU_type]

    @staticmethod
    def memory(GPU_type: 'GPUType'):
        return {
            GPUType.RTX2080_Ti: 11 * GBi,
            GPUType.Tesla_T4: 15 * GBi,
        }[GPU_type]

    @staticmethod
    def line(GPU_type: 'GPUType'):
        return {
            GPUType.RTX2080_Ti: '-',
            GPUType.Tesla_T4: '--',
        }[GPU_type]

    @staticmethod
    def marker(GPU_type: 'GPUType'):
        return {
            GPUType.RTX2080_Ti: 'o',
            GPUType.Tesla_T4: '^',
        }[GPU_type]


class ModelName(Enum):
    ResNet50 = "ResNet50"
    ResNet18 = "ResNet18"
    LSTM = "LSTM"
    BertBase = "BertBase"
    MobileNet = "MobileNet"
    YoloV5S = "YoloV5S"


class JobSpec:
    def __init__(self, job_ID: str, model_name: ModelName, batch_size: int, submit_time: int, plan_GPU: int,
                 run_time: int, total_iterations: float):
        self.job_ID: str = job_ID
        self.model_name: ModelName = model_name
        self.batch_size: int = batch_size
        self.submit_time: int = submit_time
        self.plan_GPU: int = plan_GPU
        self.run_time: int = run_time
        self.total_iterations: float = total_iterations


class Job:
    def __init__(self, job_ID: str, remaining_iterations: float, start_time: Optional[int]=None, completion_time: Optional[int]=None):
        self.job_ID: str = job_ID
        self.remaining_iterations: float = remaining_iterations
        self.start_time: Optional[int] = start_time
        self.completion_time: Optional[int] = completion_time

    def __hash__(self):
        return self.job_ID

    def __eq__(self, other):
        return self.job_ID == other.job_ID

    def __ne__(self, other):
        return self.job_ID != other.job_ID


class Task:
    def __init__(self, job_ID: str, task_idx: int):
        self.job_ID: str = job_ID
        self.task_idx: int = task_idx
        self.task_ID = f"{self.job_ID}|{self.task_idx}"

    @staticmethod
    def task_ID_to_job_ID(task_ID: str) -> str:
        return task_ID.split("|")[0]

    def __hash__(self):
        return self.task_ID

    def __eq__(self, other):
        return self.task_ID == other.task_ID

    def __ne__(self, other):
        return self.task_ID != other.task_ID


class GPU:
    def __init__(self, GPU_type: GPUType, idx: int):
        self.GPU_type: GPUType = GPU_type
        self.idx: int = idx
        self.GPU_ID = f"{self.GPU_type.name}_{self.idx}"

    def __str__(self):
        return self.GPU_ID

    def __repr__(self):
        return self.GPU_ID


class SchedulerEnum(Enum):
    MMKP = "MMKP"
    RR = "RoundRobin"
    BinPacking = "BinPacking"
    Tiresias = "Tiresias"
    Themis = "Themis"
    Optimus = "Optimus"


class SolverEnum(Enum):
    MMKP = "MMKP"
    RoundRobin = "RoundRobin"


class ProfitEnum(Enum):
    ComprehensiveUtilization = "ComprehensiveUtilization"

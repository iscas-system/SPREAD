from abc import ABC, abstractmethod
from data_source import DataSource
from object import GPUType, CompCapacity, MemoryUnit, ProfitEnum
from cluster import TaskAssignment
from typing import Set, Dict


class ProfitCalculator(ABC):
    @staticmethod
    @abstractmethod
    def calculate(data_source: DataSource, job_ID: str, GPU_type: GPUType):
        ...

    @classmethod
    def calculate_jobs(cls, data_source: DataSource, job_IDs: Set[str], GPU_type: GPUType) -> Dict[str, float]:
        d: Dict[str, float] = dict()
        for job_ID in job_IDs:
            d[job_ID] = cls.calculate(data_source=data_source, job_ID=job_ID, GPU_type=GPU_type)
        return d


class ProfitComprehensiveUtilization(ProfitCalculator):
    @staticmethod
    def calculate(data_source: DataSource, job_ID: str, GPU_type: GPUType) -> float:
        job_spec = data_source.get_job_spec(job_ID)
        worker_count = 1 if job_spec.plan_GPU <= 100 else job_spec.plan_GPU // 100
        info = DataSource.get_job_exec_info(
            model_name=job_spec.model_name,
            batch_size=job_spec.batch_size,
            GPU_type=GPU_type,
            worker_count=worker_count,
            computation_proportion=100)
        memory = (1 + info.most_memory_consumption // MemoryUnit)
        total_memory = GPUType.memory(GPU_type) // MemoryUnit * worker_count
        mem_proportion = memory / total_memory
        comp_proportion = job_spec.plan_GPU / 100.
        return comp_proportion + mem_proportion


def get_profit_calculator(profit_enum: ProfitEnum):
    return {
        ProfitEnum.ComprehensiveUtilization: ProfitComprehensiveUtilization
    }[profit_enum]
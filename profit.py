from abc import ABC, abstractmethod
from data_source import DataSource
from object import GPUType, ProfitEnum, CompCapacity
from typing import Set, Dict, Optional


class ProfitCalculator(ABC):
    @staticmethod
    @abstractmethod
    def calculate(data_source: DataSource, job_ID: str, GPU_type: GPUType, job_lack_supply: Optional[Dict[str, int]]=None):
        ...

    @classmethod
    def calculate_jobs(cls, data_source: DataSource, job_IDs: Set[str], GPU_type: GPUType, job_lack_supply: Optional[Dict[str, int]]=None) -> Dict[str, float]:
        d: Dict[str, float] = dict()
        for job_ID in job_IDs:
            d[job_ID] = cls.calculate(data_source=data_source, job_ID=job_ID, GPU_type=GPU_type, job_lack_supply=job_lack_supply)
        return d


class ProfitComprehensiveUtilization(ProfitCalculator):
    @staticmethod
    def calculate(data_source: DataSource, job_ID: str, GPU_type: GPUType, job_lack_supply: Optional[Dict[str, int]]=None) -> float:
        job_spec = data_source.get_job_spec(job_ID)
        worker_count = job_spec.plan_worker_count
        _, normalized_memory = data_source.get_job_task_memory(job_ID=job_ID, worker_count=worker_count)
        normalized_memory *= worker_count
        total_normalized_memory = GPUType.normalized_memory(GPU_type)
        mem_proportion = normalized_memory / total_normalized_memory
        comp_proportion = job_spec.plan_comp * worker_count / CompCapacity
        if job_lack_supply is not None and job_ID in job_lack_supply:
            lack_supply = job_lack_supply[job_ID]
            lack_supply_comp_proportion = (lack_supply * (100 // CompCapacity)) / 100.
            comp_proportion -= lack_supply_comp_proportion
        return comp_proportion + mem_proportion


def get_profit_calculator(profit_enum: ProfitEnum):
    return {
        ProfitEnum.ComprehensiveUtilization: ProfitComprehensiveUtilization
    }[profit_enum]
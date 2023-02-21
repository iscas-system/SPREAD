from abc import ABC, abstractmethod

import numpy as np

from data_source import DataSource
from object import GPUType, ProfitEnum, CompCapacity, TaskAssignment
from config import ClusterConfig, job_deploy_specs
from typing import Set, Dict, Optional


class ProfitCalculator(ABC):
    @staticmethod
    @abstractmethod
    def calculate(data_source: DataSource,
                  job_ID: str,
                  GPU_type: GPUType,
                  cluster_config: Optional[ClusterConfig]=None,
                  task_assignments: Optional[Set[TaskAssignment]]=None,
                  comp_req: Optional[int]=None,
                  worker_count: Optional[int]=None,
                  cross_node: Optional[bool]=None,
                  job_lack_supply: Optional[Dict[str, int]]=None):
        ...

    @classmethod
    def calculate_jobs(cls, data_source: DataSource,
                  job_IDs: Set[str],
                  GPU_type: GPUType,
                  cluster_config: Optional[ClusterConfig]=None,
                  job_ID_to_task_assignments: Optional[Dict[str, Optional[Set[TaskAssignment]]]]=None,
                  job_ID_to_comp_req: Optional[Dict[str, int]]=None,
                  job_ID_to_worker_count: Optional[Dict[str, int]]=None,
                  job_ID_to_cross_node: Optional[Dict[str, bool]]=None,
                  job_lack_supply: Optional[Dict[str, int]]=None) -> Dict[str, float]:
        d: Dict[str, float] = dict()
        def retrieve_from_dict(dic, k):
            return dic.get(k, None) if dic is not None else None

        for job_ID in job_IDs:
            d[job_ID] = cls.calculate(data_source=data_source,
                                      job_ID=job_ID,
                                      cluster_config=cluster_config,
                                      GPU_type=GPU_type,
                                      task_assignments=retrieve_from_dict(job_ID_to_task_assignments, job_ID),
                                      comp_req=retrieve_from_dict(job_ID_to_comp_req, job_ID),
                                      worker_count=retrieve_from_dict(job_ID_to_worker_count, job_ID),
                                      cross_node=retrieve_from_dict(job_ID_to_cross_node, job_ID),
                                      job_lack_supply=job_lack_supply)
        return d


class ProfitComprehensiveUtilization(ProfitCalculator):
    @staticmethod
    def calculate(data_source: DataSource,
                  job_ID: str,
                  GPU_type: GPUType,
                  cluster_config: Optional[ClusterConfig]=None,
                  task_assignments: Optional[Set[TaskAssignment]]=None,
                  comp_req: Optional[int]=None,
                  worker_count: Optional[int]=None,
                  cross_node: Optional[bool]=None,
                  job_lack_supply: Optional[Dict[str, int]]=None) -> float:
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


class ProfitThroughput(ProfitCalculator):
    @staticmethod
    def calculate(data_source: DataSource,
                  job_ID: str,
                  GPU_type: GPUType,
                  cluster_config: Optional[ClusterConfig]=None,
                  task_assignments: Optional[Set[TaskAssignment]]=None,
                  comp_req: Optional[int]=None,
                  worker_count: Optional[int]=None,
                  cross_node: Optional[bool]=None,
                  job_lack_supply: Optional[Dict[str, int]]=None) -> float:
        min_iteration_time_nano = np.inf
        for spec in job_deploy_specs:
            cross_node_, worker_count_ = spec
            _, iteration_time_nano = data_source.job_maximized_performance_comp(job_ID, GPU_type, worker_count_, cross_node_)
            min_iteration_time_nano = np.min(iteration_time_nano, min_iteration_time_nano)

        assert (task_assignments is not None and cluster_config is not None)\
               or (comp_req is not None and worker_count is not None and cross_node is not None)

        if cluster_config is not None and task_assignments is not None:
            GPU_ID_to_node_id = cluster_config.GPU_ID_to_node_id
            node_ids = set()
            comp_req = None
            for task_assignment in task_assignments:
                node_ids.add(GPU_ID_to_node_id[task_assignment.GPU_ID])
                if comp_req is None:
                    comp_req = task_assignment.comp_req
                assert comp_req == task_assignment.comp_req
            assert comp_req is not None
            cross_node = len(node_ids) > 1

        iteration_time_nano = data_source.job_iteration_time_nano(job_ID=job_ID, GPU_type=GPU_type, comp_req=comp_req, worker_count=len(task_assignments), cross_node=cross_node)
        throughput_ratio = 1. * min_iteration_time_nano / iteration_time_nano
        return throughput_ratio

def get_profit_calculator(profit_enum: ProfitEnum=ProfitEnum.Throughput):
    return {
        ProfitEnum.ComprehensiveUtilization: ProfitComprehensiveUtilization,
        ProfitEnum.Throughput: ProfitThroughput
    }[profit_enum]
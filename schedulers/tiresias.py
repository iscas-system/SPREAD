from collections import defaultdict
from typing import Tuple, Optional, Set, Dict, List, Any

from cluster import TaskAssignment, Assignments
from object import CompCapacity, GPUType, Task, PriorityType, Job
from scheduler import Scheduler
from schedulers.sorter import Sorter
import numpy as np
from collections import namedtuple


class TiresiasScheduler(Scheduler):
    def _init_config(self):
        self.GPU_type = GPUType.RTX_2080Ti
        self.job_attained_service: Dict[str, int] = defaultdict(int)
        self.last_schedule = 0

    def do_assign(self, preemptive: bool, now: int, done_jobs_between_preemption: Set[Job]) -> Tuple[Assignments, Optional[Any]]:
        GPU_ID_to_task_assignments, job_IDs = self.prepare_assign_ctx(preemptive)
        duration = now - self.last_schedule
        self.last_schedule = now
        for job_ID, task_assignments in self.cluster.assignments.job_ID_to_task_assignments.items():
            worker_count = len(task_assignments)
            self.job_attained_service[job_ID] += duration * worker_count
        JAS = namedtuple("JAS", field_names=["job_ID", "attained_service"])
        JAS_list = list()
        for job_ID in job_IDs:
            JAS_list.append(JAS(job_ID=job_ID, attained_service=self.job_attained_service[job_ID]))
        JAS_list.sort(key=lambda jas: jas.attained_service)
        JAS_list = JAS_list[:300]
        GPU_ID_comp_mem_type = namedtuple(typename="GPU_ID_comp", field_names=["GPU_ID", "comp", "mem"])
        GPU_mem = GPUType.normalized_memory(GPU_type=self.GPU_type)
        for jas in JAS_list:
            job_ID = jas.job_ID
            GPU_ID_to_remain_comp_mem = self.GPU_remain_comp_mem(GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
            job_spec = self.data_source.get_job_spec(job_ID)
            comp_req = self.data_source.job_maximized_performance_comp(job_ID=job_ID, worker_count=job_spec.plan_worker_count, GPU_type=self.GPU_type, cross_node=False)
            remain_GPU_ID_comp_mem_list: List[GPU_ID_comp_mem_type] = list()
            for GPU_ID in self.cluster.cluster_config.GPU_IDs:
                remain_comp_mem = GPU_ID_to_remain_comp_mem[GPU_ID]
                comp, mem = remain_comp_mem
                remain_GPU_ID_comp_mem_list.append(GPU_ID_comp_mem_type(GPU_ID, comp, mem))
            remain_GPU_ID_comp_mem_list.sort(key=lambda t: t[0], reverse=True)
            window_size = job_spec.plan_worker_count
            _, task_mem = self.data_source.get_job_task_memory(job_ID, job_spec.plan_worker_count)
            comp_enough_GPU_ID_com_mem_slice_list: List[Tuple[List[GPU_ID_comp_mem_type], float]] = list()
            for i in range(len(remain_GPU_ID_comp_mem_list) - window_size + 1):
                comp_enough = True
                slice_GPU_ID_comp_mem = remain_GPU_ID_comp_mem_list[i: i + window_size]
                slice_total_resource = 0
                for sliding_idx, item in enumerate(slice_GPU_ID_comp_mem):
                    if item.comp < comp_req:
                        comp_enough = False
                        continue
                    slice_total_resource += item.comp / CompCapacity
                    if item.mem < task_mem:
                        comp_enough = False
                        continue
                    # slice_total_resource += item.mem / GPU_mem
                    break
                if not comp_enough:
                    continue
                comp_enough_GPU_ID_com_mem_slice_list.append((slice_GPU_ID_comp_mem, slice_total_resource))
            comp_enough_GPU_ID_com_mem_slice_list.sort(key=lambda slice_tuple: slice_tuple[1])
            if len(comp_enough_GPU_ID_com_mem_slice_list) == 0:
                continue
            slice_GPU_ID_comp_mem = comp_enough_GPU_ID_com_mem_slice_list[0][0]
            for task_idx in range(job_spec.plan_worker_count):
                GPU_ID_comp_mem = slice_GPU_ID_comp_mem[task_idx]
                GPU_ID, _, _ = GPU_ID_comp_mem
                task_assignment = TaskAssignment(GPU_ID=GPU_ID,
                                                 GPU_type=self.GPU_type,
                                                 task=Task(job_ID=job_ID, task_idx=task_idx),
                                                 comp_req=job_spec.plan_comp,
                                                 memory=task_mem)
                GPU_ID_to_task_assignments[GPU_ID].add(task_assignment)
        assignments = Assignments.from_GPU_ID_to_task_assignments(
            cluster_config=self.cluster.cluster_config,
            GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        return assignments, None

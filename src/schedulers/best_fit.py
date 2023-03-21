from collections import namedtuple
from collections import namedtuple
from typing import Tuple, Optional, Set, List, Any, Dict

import config
from cluster import TaskAssignment, Assignments
from object import CompCapacity, GPUType, Task, PriorityType, Job
from scheduler import Scheduler
from schedulers.sorter import Sorter


class BestFitScheduler(Scheduler):
    def _init_config(self):
        self.GPU_type = GPUType.RTX_2080Ti
        ...

    def do_assign(self, preemptive: bool, now: int, done_jobs_between_preemption: Set[Job]) -> Tuple[
        Assignments, Optional[Any]]:
        GPU_ID_to_task_assignments, job_IDs = self.prepare_assign_ctx(preemptive)
        job_IDs = Sorter.sort(jobs=[self.cluster.get_job(job_ID) for job_ID in job_IDs], data_source=self.data_source,
                              priority_type=PriorityType.FCFS)
        return BestFitScheduler.best_fit_assign(self.cluster.cluster_config, GPU_ID_to_task_assignments, job_IDs), None

    @staticmethod
    def best_fit_assign(scheduler: Scheduler,
                        GPU_type: GPUType,
                        GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]],
                        job_IDs: List[str]) -> Assignments:
        GPU_ID_comp_mem_type = namedtuple(typename="GPU_ID_comp", field_names=["GPU_ID", "comp", "mem"])
        GPU_mem = GPUType.normalized_memory(GPU_type=GPU_type)
        job_IDs = job_IDs[:128]
        for job_ID in job_IDs:
            _, spec = scheduler.data_source.job_maximized_performance(job_ID=job_ID, GPU_type=GPU_type)
            cross_node, worker_count = spec
            GPU_ID_to_remain_comp_mem = scheduler.GPU_remain_comp_mem(GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
            remain_GPU_ID_comp_mem_list: List[GPU_ID_comp_mem_type] = list()
            for GPU_ID in scheduler.cluster.cluster_config.GPU_IDs:
                remain_comp_mem = GPU_ID_to_remain_comp_mem[GPU_ID]
                comp, mem = remain_comp_mem
                remain_GPU_ID_comp_mem_list.append(GPU_ID_comp_mem_type(GPU_ID, comp, mem))
            remain_GPU_ID_comp_mem_list.sort(key=lambda t: t[0], reverse=True)
            window_size = worker_count
            task_comp, _ = scheduler.data_source.job_maximized_performance_comp(job_ID=job_ID, GPU_type=GPU_type,
                                                                           worker_count=worker_count, cross_node=cross_node)
            _, task_mem = scheduler.data_source.get_job_task_memory(job_ID, worker_count=worker_count)
            comp_enough_GPU_ID_com_mem_slice_list: List[Tuple[List[GPU_ID_comp_mem_type], float]] = list()
            for i in range(len(remain_GPU_ID_comp_mem_list) - window_size + 1):
                comp_enough = True
                slice_GPU_ID_comp_mem = remain_GPU_ID_comp_mem_list[i: i + window_size]
                slice_total_resource = 0
                for sliding_idx, item in enumerate(slice_GPU_ID_comp_mem):
                    if item.comp < task_comp:
                        comp_enough = False
                        continue
                    slice_total_resource += item.comp / CompCapacity
                    if item.mem < task_mem:
                        comp_enough = False
                        continue
                    break
                if not comp_enough:
                    continue
                comp_enough_GPU_ID_com_mem_slice_list.append((slice_GPU_ID_comp_mem, slice_total_resource))
            comp_enough_GPU_ID_com_mem_slice_list.sort(key=lambda slice_tuple: slice_tuple[1])
            if len(comp_enough_GPU_ID_com_mem_slice_list) == 0:
                continue
            slice_GPU_ID_comp_mem = comp_enough_GPU_ID_com_mem_slice_list[0][0]
            for task_idx in range(worker_count):
                GPU_ID_comp_mem = slice_GPU_ID_comp_mem[task_idx]
                GPU_ID, _, _ = GPU_ID_comp_mem
                task_assignment = TaskAssignment(GPU_ID=GPU_ID,
                                                 GPU_type=GPU_type,
                                                 task=Task(job_ID=job_ID, task_idx=task_idx),
                                                 comp_req=task_comp,
                                                 memory=task_mem)
                GPU_ID_to_task_assignments[GPU_ID].add(task_assignment)
        assignments = Assignments.from_GPU_ID_to_task_assignments(
            scheduler.cluster.cluster_config,
            GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        # oversupplied_assignments = assignments.supplement_over_supply()
        return assignments

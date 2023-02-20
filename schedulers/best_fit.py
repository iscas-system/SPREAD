from collections import defaultdict
from collections import namedtuple
from typing import Tuple, Optional, Set, Dict, List, Any

from cluster import TaskAssignment, Assignments
from object import CompCapacity, GPUType, Task, PriorityType, Job
from scheduler import Scheduler
from schedulers.sorter import Sorter


class BestFitScheduler(Scheduler):
    def _init_config(self):
        self.GPU_type = GPUType.RTX_2080Ti
        ...

    def GPU_remain_comp_mem(self, GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]]) -> Dict[
        str, Tuple[int, int]]:
        GPU_ID_to_remain_comp_mem: Dict[str, Tuple[int, int]] = dict()
        for GPU_ID in self.cluster.GPU_IDs:
            GPU_ID_to_remain_comp_mem[GPU_ID] = CompCapacity, GPUType.normalized_memory(
                self.cluster.GPU_ID_to_GPU_type[GPU_ID])
        for GPU_ID in self.cluster.GPU_IDs:
            GPU_mem = GPUType.normalized_memory(
                GPU_type=self.cluster.GPU_ID_to_GPU_type[GPU_ID])
            task_assignments = GPU_ID_to_task_assignments[GPU_ID]
            total_comp = 0
            total_mem = 0
            for task_assignment in task_assignments:
                total_comp += task_assignment.comp_req
                total_mem += task_assignment.memory
            remain_comp = CompCapacity - total_comp
            remain_mem = GPU_mem - total_mem
            GPU_ID_to_remain_comp_mem[GPU_ID] = remain_comp, remain_mem
        return GPU_ID_to_remain_comp_mem

    def do_assign(self, preemptive: bool, now: int, done_jobs_between_preemption: Set[Job]) -> Tuple[
        Assignments, Optional[Any]]:
        GPU_ID_to_task_assignments, job_IDs = self.prepare_assign_ctx(preemptive)
        job_IDs = Sorter.sort(jobs=[self.cluster.get_job(job_ID) for job_ID in job_IDs], data_source=self.data_source,
                              priority_type=PriorityType.FCFS)
        GPU_ID_comp_mem_type = namedtuple(typename="GPU_ID_comp", field_names=["GPU_ID", "comp", "mem"])
        GPU_mem = GPUType.normalized_memory(GPU_type=self.GPU_type)
        job_IDs = job_IDs[:300]
        for job_ID in job_IDs:
            GPU_ID_to_remain_comp_mem = self.GPU_remain_comp_mem(GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
            job_spec = self.data_source.get_job_spec(job_ID)
            remain_GPU_ID_comp_mem_list: List[GPU_ID_comp_mem_type] = list()
            for GPU_ID in self.cluster.GPU_IDs:
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
                    if item.comp < job_spec.plan_comp:
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
            self.cluster.cluster_config,
            GPU_ID_to_GPU_type=defaultdict(lambda: self.GPU_type),
            GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        # oversupplied_assignments = assignments.supplement_over_supply()
        return assignments, None

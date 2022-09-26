from collections import defaultdict
from typing import Tuple, Optional, Set, Dict, List, Any

import numpy as np

from cluster import TaskAssignment, Assignments
from data_source import DataSource
from object import CompCapacity, GPUType, Task, Job
from scheduler import Scheduler
from schedulers.sorter import Sorter
from collections import namedtuple


class GavelScheduler(Scheduler):
    def _init_config(self):
        self.strict = self.config.get("strict", True)
        self.GPU_type = GPUType.RTX_2080Ti

    def do_assign(self, preemptive: bool, now: int, done_jobs_between_preemption: Set[Job]) -> Tuple[Assignments, Optional[Any]]:
        GPU_ID_to_task_assignments, job_IDs = self.prepare_assign_ctx(preemptive)

        GPU_ID_comp_mem_type = namedtuple(typename="GPU_ID_comp", field_names=["GPU_ID", "comp", "mem"])
        GPU_mem = GPUType.normalized_memory(GPU_type=self.GPU_type)
        job_IDs = job_IDs[:100]
        assigned_job_IDs = set()
        while True:
            job_ID_to_comp_enough_GPU_ID_com_mem_slice = dict()
            for job_ID in job_IDs:
                if job_ID in assigned_job_IDs:
                    continue
                GPU_ID_to_remain_comp_mem = self.GPU_remain_comp_mem(GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
                job_spec = self.data_source.get_job_spec(job_ID)
                remain_GPU_ID_comp_mem_list: List[GPU_ID_comp_mem_type] = list()
                for GPU_ID in self.cluster.GPU_IDs:
                    remain_comp_mem = GPU_ID_to_remain_comp_mem[GPU_ID]
                    comp, mem = remain_comp_mem
                    remain_GPU_ID_comp_mem_list.append(GPU_ID_comp_mem_type(GPU_ID, comp, mem))
                if len(remain_GPU_ID_comp_mem_list) == 0:
                    break
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
                        if item.mem < task_mem:
                            comp_enough = False
                            continue
                        slice_total_resource += item.comp / CompCapacity + item.mem / GPU_mem
                        break
                    if not comp_enough:
                        continue
                    comp_enough_GPU_ID_com_mem_slice_list.append((slice_GPU_ID_comp_mem, slice_total_resource))
                comp_enough_GPU_ID_com_mem_slice_list.sort(key=lambda slice_tuple: slice_tuple[1])
                if len(comp_enough_GPU_ID_com_mem_slice_list) == 0:
                    continue
                job_ID_to_comp_enough_GPU_ID_com_mem_slice[job_ID] = comp_enough_GPU_ID_com_mem_slice_list[0]

            if len(job_ID_to_comp_enough_GPU_ID_com_mem_slice) == 0:
                break

            min_total_resource = np.inf
            min_frag_job_ID = None
            for job_ID, com_mem_slice in job_ID_to_comp_enough_GPU_ID_com_mem_slice.items():
                total_resource = com_mem_slice[1]
                if total_resource < min_total_resource:
                    min_total_resource = total_resource
                    min_frag_job_ID = job_ID
            assigned_job_IDs.add(min_frag_job_ID)
            slice_GPU_ID_comp_mem = job_ID_to_comp_enough_GPU_ID_com_mem_slice[min_frag_job_ID][0]
            job_spec = self.data_source.get_job_spec(min_frag_job_ID)
            _, task_mem = self.data_source.get_job_task_memory(min_frag_job_ID, job_spec.plan_worker_count)
            for task_idx in range(job_spec.plan_worker_count):
                GPU_ID_comp_mem = slice_GPU_ID_comp_mem[task_idx]
                GPU_ID, _, _ = GPU_ID_comp_mem
                task_assignment = TaskAssignment(GPU_ID=GPU_ID,
                                                 GPU_type=self.GPU_type,
                                                 task=Task(job_ID=min_frag_job_ID, task_idx=task_idx),
                                                 comp_req=job_spec.plan_comp,
                                                 memory=task_mem)
                GPU_ID_to_task_assignments[GPU_ID].add(task_assignment)

        assignments = Assignments.from_GPU_ID_to_task_assignments(GPU_ID_to_GPU_type=defaultdict(lambda: self.GPU_type),
                                                                  GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        # oversupplied_assignments = assignments.supplement_over_supply()
        return assignments, None

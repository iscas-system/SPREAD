from collections import defaultdict
from collections import namedtuple
from typing import Tuple, Optional, Set, List, Any

from cluster import TaskAssignment, Assignments
from object import GPUType, Task, PriorityType, Job
from scheduler import Scheduler
from schedulers.sorter import Sorter


class KubeShareScheduler(Scheduler):
    def _init_config(self):
        self.GPU_type = GPUType.RTX_2080Ti
        ...

    def do_assign(self, preemptive: bool, now: int, done_jobs_between_preemption: Set[Job]) -> Tuple[
        Assignments, Optional[Any]]:
        GPU_ID_to_task_assignments, job_IDs = self.prepare_assign_ctx(preemptive)
        job_IDs = Sorter.sort(jobs=[self.cluster.get_job(job_ID) for job_ID in job_IDs], data_source=self.data_source,
                              priority_type=PriorityType.FCFS)
        GPU_ID_comp_mem_type = namedtuple(typename="GPU_ID_comp", field_names=["GPU_ID", "comp", "mem"])
        job_IDs = job_IDs[:400]
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
            for i in range(len(remain_GPU_ID_comp_mem_list) - window_size + 1):
                resource_enough = True
                slice_GPU_ID_comp_mem = remain_GPU_ID_comp_mem_list[i: i + window_size]
                for sliding_idx, item in enumerate(slice_GPU_ID_comp_mem):
                    if item.comp < job_spec.plan_comp:
                        resource_enough = False
                        continue
                    if item.mem < task_mem:
                        resource_enough = False
                        continue
                    break
                if not resource_enough:
                    continue

                for task_idx in range(job_spec.plan_worker_count):
                    GPU_ID_comp_mem = slice_GPU_ID_comp_mem[task_idx]
                    GPU_ID, _, _ = GPU_ID_comp_mem
                    task_assignment = TaskAssignment(GPU_ID=GPU_ID,
                                                     GPU_type=self.GPU_type,
                                                     task=Task(job_ID=job_ID, task_idx=task_idx),
                                                     comp_req=job_spec.plan_comp,
                                                     memory=task_mem)
                    GPU_ID_to_task_assignments[GPU_ID].add(task_assignment)
                if resource_enough:
                    break
        assignments = Assignments.from_GPU_ID_to_task_assignments(
            cluster_config=self.cluster.cluster_config,
            GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        # oversupplied_assignments = assignments.supplement_over_supply()
        return assignments, None

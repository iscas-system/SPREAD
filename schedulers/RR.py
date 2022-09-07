from typing import Tuple, Optional, Set, Dict, List, Any

from scheduler import Scheduler
from schedulers.sorter import Sorter
from object import PriorityType, CompCapacity, GPUType, Task
from cluster import TaskAssignment, Assignments
from collections import defaultdict


class RRScheduler(Scheduler):
    def _init_config(self):
        self.strict = self.config.get("strict", True)

    def do_assign(self, preemptive: bool) -> Tuple[Assignments, Optional[Any]]:
        jobs = self.cluster.jobs
        job_IDs = Sorter.sort(jobs=jobs.values(), data_source=self.data_source, priority_type=self.priority_type)
        curr_GPU_ID_idx = -1
        GPU_IDs = sorted(self.cluster.GPU_IDs)
        if not preemptive:
            GPU_ID_to_task_assignments = self.cluster.assignments.GPU_ID_to_task_assignments
        else:
            GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]] = defaultdict(set)


        for job_ID in job_IDs:
            curr_GPU_ID_idx = self.assign_job(job_ID=job_ID,
                            curr_GPU_ID_idx=curr_GPU_ID_idx,
                            GPU_IDs=GPU_IDs,
                            GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        assignments = Assignments.from_GPU_ID_to_task_assignments(GPU_ID_to_GPU_type=self.cluster.GPU_ID_to_GPU_type,
                                                    GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        oversupplied_assignments = assignments.supplement_over_supply()

        return oversupplied_assignments, None


    def assign_job(self, job_ID: str, curr_GPU_ID_idx: int, GPU_IDs: List[str], GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]]) -> int:
        original_GPU_ID_idx = curr_GPU_ID_idx
        job_spec = self.data_source.get_job_spec(job_ID=job_ID)
        worker_count = job_spec.plan_worker_count
        GPU_ID_tried = set()
        GPU_IDs_selected = set()
        GPU_type = None
        comp_requirement = job_spec.plan_comp
        mem_requirement = None
        capacity_allowed = False
        while True:
            curr_GPU_ID_idx += 1
            first_GPU = GPU_IDs[curr_GPU_ID_idx % len(GPU_IDs)]
            if first_GPU in GPU_ID_tried:
                break
            GPU_ID_tried.add(first_GPU)

            GPU_IDs_selected = set()
            for i in range(worker_count):
                GPU_IDs_selected.add(GPU_IDs[(curr_GPU_ID_idx + i) % len(GPU_IDs)])
            GPU_ID_to_GPU_type = dict()
            for GPU_ID in GPU_IDs_selected:
                GPU_ID_to_GPU_type[GPU_ID] = self.cluster.get_GPU_by_ID(GPU_ID).GPU_type
            GPU_type = set(GPU_ID_to_GPU_type.values())
            if len(GPU_type) != 1:
                continue
            GPU_type = next(iter(GPU_type))
            _, mem_requirement = self.data_source.get_job_task_memory(job_ID=job_ID, worker_count=worker_count)
            for GPU_ID in GPU_IDs_selected:
                task_assignments = GPU_ID_to_task_assignments[GPU_ID]
                total_comp = 0
                total_mem = 0
                for task_assignment in task_assignments:
                    total_comp += task_assignment.comp_req
                    total_mem += task_assignment.memory
                if comp_requirement + total_comp > CompCapacity:
                    break
                if mem_requirement + total_mem > GPUType.normalized_memory(GPU_type):
                    break
                capacity_allowed = True
            if capacity_allowed:
                break
        if not capacity_allowed:
            return original_GPU_ID_idx
        for GPU_ID in GPU_IDs_selected:
            for i in range(worker_count):
                task_assignment = TaskAssignment(GPU_ID=GPU_ID,
                                                 GPU_type=GPU_type,
                                                 task=Task(job_ID=job_ID, task_idx=i),
                                                 comp_req=comp_requirement,
                                                 memory=mem_requirement)
                GPU_ID_to_task_assignments[GPU_ID].add(task_assignment)
        return curr_GPU_ID_idx

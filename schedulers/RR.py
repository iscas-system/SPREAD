from collections import defaultdict
from typing import Tuple, Optional, Set, Dict, List, Any

import numpy as np

from cluster import TaskAssignment, Assignments
from object import CompCapacity, GPUType, Task
from scheduler import Scheduler
from schedulers.sorter import Sorter

class RRScheduler(Scheduler):
    def _init_config(self):
        self.strict = self.config.get("strict", True)

    def do_assign(self, preemptive: bool) -> Tuple[Assignments, Optional[Any]]:
        GPU_IDs = sorted(self.cluster.GPU_IDs)
        if not preemptive:
            GPU_ID_to_task_assignments = self.cluster.assignments.clone().GPU_ID_to_task_assignments
        else:
            GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]] = defaultdict(set)
        job_IDs = set()
        if not preemptive:
            for job in self.cluster.jobs.values():
                if job.job_ID not in self.cluster.assignments.job_ID_to_task_assignments:
                    job_IDs.add(job.job_ID)
        else:
            job_IDs = list(self.cluster.jobs.keys())
        job_IDs = Sorter.sort(jobs=[self.cluster.jobs[job_ID] for job_ID in job_IDs], data_source=self.data_source, priority_type=self.priority_type)
        most_remain_GPU_ID = self.most_remain_GPU_ID(GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        curr_GPU_ID_idx = GPU_IDs.index(most_remain_GPU_ID) - 1

        if self.strict:
            assign_job = self.strict_assign_job
        else:
            assign_job = self.best_effort_assign_job

        for job_ID in job_IDs:
            curr_GPU_ID_idx, success = assign_job(job_ID=job_ID,
                                         curr_GPU_ID_idx=curr_GPU_ID_idx,
                                         GPU_IDs=GPU_IDs,
                                         GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        assignments = Assignments.from_GPU_ID_to_task_assignments(GPU_ID_to_GPU_type=self.cluster.GPU_ID_to_GPU_type,
                                                                  GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        oversupplied_assignments = assignments.supplement_over_supply()

        return oversupplied_assignments, None

    def most_remain_GPU_ID(self, GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]]) -> str:
        most_remain_resource = 0
        most_remain_GPU_ID = None
        for GPU_ID in self.cluster.GPU_IDs:
            task_assignments = GPU_ID_to_task_assignments[GPU_ID]
            total_comp = 0
            total_mem = 0
            for task_assignment in task_assignments:
                total_comp += task_assignment.comp_req
                total_mem += task_assignment.memory
            normalized_remain_comp = total_comp / CompCapacity
            normalized_remain_mem = total_mem / GPUType.normalized_memory(GPU_type=self.cluster.GPU_ID_to_GPU_type[GPU_ID])
            normalized_remain_resource = normalized_remain_comp + normalized_remain_mem
            if normalized_remain_resource >= most_remain_resource:
                most_remain_resource = normalized_remain_resource
                most_remain_GPU_ID = GPU_ID
        return most_remain_GPU_ID

    def strict_assign_job(self, job_ID: str, curr_GPU_ID_idx: int, GPU_IDs: List[str],
                          GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]]) -> Tuple[int, bool]:
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
            return original_GPU_ID_idx, False
        for GPU_ID in GPU_IDs_selected:
            for i in range(worker_count):
                task_assignment = TaskAssignment(GPU_ID=GPU_ID,
                                                 GPU_type=GPU_type,
                                                 task=Task(job_ID=job_ID, task_idx=i),
                                                 comp_req=comp_requirement,
                                                 memory=mem_requirement)
                GPU_ID_to_task_assignments[GPU_ID].add(task_assignment)
        return curr_GPU_ID_idx, True

    def best_effort_assign_job(self, job_ID: str, curr_GPU_ID_idx: int, GPU_IDs: List[str],
                               GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]]) -> Tuple[int, bool]:
        strict_assign_GPU_ID_idx, success = self.strict_assign_job(job_ID, curr_GPU_ID_idx, GPU_IDs, GPU_ID_to_task_assignments)
        if success:
            return strict_assign_GPU_ID_idx, True

        original_GPU_ID_idx = curr_GPU_ID_idx
        job_spec = self.data_source.get_job_spec(job_ID=job_ID)
        worker_count = job_spec.plan_worker_count
        GPU_ID_tried = set()
        GPU_IDs_selected = set()
        GPU_type = None
        comp_requirement = job_spec.plan_comp
        mem_requirement = None
        capacity_allowed = False
        remain_comp = None
        while not capacity_allowed:
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
            not_enough_resource = False
            minimum_remain_comp = np.inf
            for GPU_ID in GPU_IDs_selected:
                task_assignments = GPU_ID_to_task_assignments[GPU_ID]
                total_comp = 0
                total_mem = 0
                for task_assignment in task_assignments:
                    total_comp += task_assignment.comp_req
                    total_mem += task_assignment.memory
                if mem_requirement + total_mem > GPUType.normalized_memory(GPU_type):
                    not_enough_resource = True
                    break
                if total_comp == CompCapacity:
                    not_enough_resource = True
                    break
                minimum_remain_comp = min(CompCapacity - total_comp, minimum_remain_comp)
            if not_enough_resource:
                continue
            capacity_allowed = True
            remain_comp = minimum_remain_comp
            assert remain_comp < comp_requirement
        if not capacity_allowed:
            return original_GPU_ID_idx, False
        assert remain_comp is not None
        for GPU_ID in GPU_IDs_selected:
            for i in range(worker_count):
                task_assignment = TaskAssignment(GPU_ID=GPU_ID,
                                                 GPU_type=GPU_type,
                                                 task=Task(job_ID=job_ID, task_idx=i),
                                                 comp_req=remain_comp,
                                                 memory=mem_requirement)
                GPU_ID_to_task_assignments[GPU_ID].add(task_assignment)
        return curr_GPU_ID_idx, True

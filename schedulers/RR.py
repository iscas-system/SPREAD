from collections import defaultdict
from typing import Tuple, Optional, Set, Dict, List, Any

import numpy as np

import config
from cluster import TaskAssignment, Assignments
from data_source import DataSource
from object import CompCapacity, GPUType, Task, Job
from scheduler import Scheduler
from schedulers.sorter import Sorter


class RRScheduler(Scheduler):
    def _init_config(self):
        self.GPU_type = GPUType.RTX_2080Ti

    def do_assign(self, preemptive: bool, now: int, done_jobs_between_preemption: Set[Job]) -> Tuple[
        Assignments, Optional[Any]]:
        GPU_IDs = sorted(self.cluster.cluster_config.GPU_IDs)
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
        job_IDs = Sorter.sort(jobs=[self.cluster.jobs[job_ID] for job_ID in job_IDs], data_source=self.data_source,
                              priority_type=self.priority_type)
        job_IDs = job_IDs[:300]
        assignments = RRScheduler.assign_jobs(self.cluster.cluster_config, self.data_source, job_IDs,
                                              GPU_IDs, self.GPU_type, GPU_ID_to_task_assignments)
        # oversupplied_assignments = assignments.supplement_over_supply()

        return assignments, None

    @staticmethod
    def assign_jobs(cluster_config: config.ClusterConfig,
                    data_source: DataSource,
                    job_IDs: List[str],
                    GPU_IDs: List[str],
                    GPU_type: GPUType,
                    GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]]):
        most_remain_GPU_ID = RRScheduler.most_remain_GPU_ID(GPU_IDs=GPU_IDs, GPU_type=GPU_type,
                                                            GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        curr_GPU_ID_idx = GPU_IDs.index(most_remain_GPU_ID) - 1
        assign_job = RRScheduler.assign_job
        for job_ID in job_IDs:
            curr_GPU_ID_idx, success = assign_job(data_source=data_source,
                                                  GPU_type=GPU_type,
                                                  job_ID=job_ID,
                                                  curr_GPU_ID_idx=curr_GPU_ID_idx,
                                                  GPU_IDs=GPU_IDs,
                                                  GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        assignments = Assignments.from_GPU_ID_to_task_assignments(
            cluster_config=cluster_config,
            GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        return assignments

    @staticmethod
    def most_remain_GPU_ID(GPU_IDs: List[str], GPU_type: GPUType,
                           GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]]) -> str:
        most_remain_resource = 0
        most_remain_GPU_ID = None
        for GPU_ID in GPU_IDs:
            task_assignments = GPU_ID_to_task_assignments[GPU_ID]
            total_comp = 0
            total_mem = 0
            GPU_mem = GPUType.normalized_memory(GPU_type=GPU_type)
            for task_assignment in task_assignments:
                total_comp += task_assignment.comp_req
                total_mem += task_assignment.memory
            normalized_remain_comp = (CompCapacity - total_comp) / CompCapacity
            normalized_remain_mem = (GPU_mem - total_mem) / GPU_mem
            normalized_remain_resource = normalized_remain_comp + normalized_remain_mem
            if normalized_remain_resource >= most_remain_resource:
                most_remain_resource = normalized_remain_resource
                most_remain_GPU_ID = GPU_ID
        return most_remain_GPU_ID

    @staticmethod
    def assign_job(data_source: DataSource,
                   GPU_type: GPUType,
                   job_ID: str,
                   curr_GPU_ID_idx: int,
                   GPU_IDs: List[str],
                   GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]]) -> Tuple[int, bool]:
        original_GPU_ID_idx = curr_GPU_ID_idx
        job_spec = data_source.get_job_spec(job_ID=job_ID)
        worker_count = job_spec.plan_worker_count
        GPU_ID_tried = set()
        GPU_IDs_selected = set()
        comp_requirement, _ = data_source.job_maximized_performance_comp(job_ID=job_ID, GPU_type=GPU_type, worker_count=worker_count, cross_node=False)
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
            _, mem_requirement = data_source.get_job_task_memory(job_ID=job_ID, worker_count=worker_count)
            GPU_IDs_capacity_allowed = defaultdict()
            for GPU_ID in GPU_IDs_selected:
                task_assignments = GPU_ID_to_task_assignments[GPU_ID]
                total_comp = 0
                total_mem = 0
                for task_assignment in task_assignments:
                    total_comp += task_assignment.comp_req
                    total_mem += task_assignment.memory
                if comp_requirement + total_comp > CompCapacity:
                    GPU_IDs_capacity_allowed[GPU_ID] = False
                    break
                if mem_requirement + total_mem > GPUType.normalized_memory(GPU_type):
                    GPU_IDs_capacity_allowed[GPU_ID] = False
                    break
                GPU_IDs_capacity_allowed[GPU_ID] = True
            if set(GPU_IDs_capacity_allowed.values()) == {True}:
                capacity_allowed = True
                break
        if not capacity_allowed:
            return original_GPU_ID_idx, False
        assert len(GPU_IDs_selected) == worker_count
        for i, GPU_ID in enumerate(GPU_IDs_selected):
            task_assignment = TaskAssignment(GPU_ID=GPU_ID,
                                             GPU_type=GPU_type,
                                             task=Task(job_ID=job_ID, task_idx=i),
                                             comp_req=comp_requirement,
                                             memory=mem_requirement)
            GPU_ID_to_task_assignments[GPU_ID].add(task_assignment)

        return curr_GPU_ID_idx, True

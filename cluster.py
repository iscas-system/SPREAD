from collections import defaultdict
from typing import Dict, List, DefaultDict, Optional, Set, Union, Tuple

from config import ClusterConfig, get_config
from object import GPU, GPUType, Job, Task
from data_source import DataSource
from profit import ProfitCalculator

import numpy as np


class Cluster:
    def __init__(self, cluster_config: ClusterConfig):
        self.GPUs: DefaultDict[GPUType, List[GPU]] = defaultdict(list)
        self.GPU_IDs: List[str] = list()
        self.GPU_ID_to_GPU: Dict[str, GPU] = dict()
        for GPU_type, count in cluster_config.GPUs:
            for i in range(count):
                g = GPU(GPU_type, i)
                self.GPUs[GPU_type].append(g)
                self.GPU_IDs.append(g.GPU_ID)
                self.GPU_ID_to_GPU[g.GPU_ID] = g

        self.assignments: Assignments = Assignments()
        self.jobs: Dict[str, Job] = dict()
        self.done_jobs: Dict[str, Job] = dict()

    def get_GPU_by_ID(self, GPU_ID: str) -> GPU:
        return self.GPU_ID_to_GPU[GPU_ID]

    def get_undone_job(self, job_ID: str) -> Job:
        return self.jobs[job_ID]

    def get_job(self, job_ID: str) -> Job:
        if job_ID in self.jobs:
            return self.jobs[job_ID]
        return self.done_jobs[job_ID]

    def submit(self, job: Job):
        assert job.job_ID not in self.jobs
        self.jobs[job.job_ID] = job

    def done(self, job_ID: str, now: int):
        assert job_ID in self.jobs
        assert self.jobs[job_ID].completion_time is None
        self.jobs[job_ID].completion_time = now
        self.jobs[job_ID].remaining_iterations = 0

    def ensure_start(self, job_ID: str, now: int):
        assert job_ID in self.jobs
        if self.jobs[job_ID].start_time is None:
            self.jobs[job_ID].start_time = now

    def add_preemptive_overhead(self, job_ID: str, overhead_iterations: float):
        self.jobs[job_ID].remaining_iterations += overhead_iterations

    def calc_profits(self, data_source: DataSource, profit_calculator: ProfitCalculator) -> float:
        total_profit = 0
        for GPU_ID, job_ID_to_task_assignments in self.assignments.GPU_to_task_assignments:
            GPU_type = self.get_GPU_by_ID(GPU_ID=GPU_ID).GPU_type
            p = profit_calculator.calculate_jobs(data_source=data_source, job_IDs=job_ID_to_task_assignments.keys(), GPU_type=GPU_type)
            total_profit += p
        return total_profit

class TaskAssignment:
    def __init__(self, GPU_type: GPUType, task: Task, comp_proportion: int, memory: int):
        self.GPU_type: GPUType = GPU_type
        self.task: Task = task
        self.comp_proportion: int = comp_proportion
        self.memory: int = memory

    def __hash__(self):
        return self.task

    def __eq__(self, other):
        return self.task == other.task

    def __ne__(self, other):
        return self.task != other.task


class Assignments:
    def __init__(self, GPU_to_task_assignments: Optional[Dict[str, Dict[str, Set[TaskAssignment]]]] = None):
        if GPU_to_task_assignments is None:
            GPU_to_task_assignments = dict()
        self.GPU_to_task_assignments: Dict[str, Dict[str, Set[TaskAssignment]]] = GPU_to_task_assignments
        self.__init_views()

    def __init_views(self):
        self.job_ID_to_task_assignments: Dict[str, Set[TaskAssignment]] = dict()
        for GPU_type, job_ID_task_assignments in self.GPU_to_task_assignments.items():
            for job_ID, task_assignments in job_ID_task_assignments.items():
                self.job_ID_to_task_assignments[job_ID] = task_assignments

    def to_solver_assignments(self) -> Dict[str, Set[str]]:
        d: Dict[str, Set[str]] = defaultdict(set)
        for g, job_to_task_assignments in self.GPU_to_task_assignments.items():
            for job, task_assignments in job_to_task_assignments.items():
                task_IDs_set = {task_assignment.task.task_ID for task_assignment in task_assignments}
                d[g] += task_IDs_set
        return d

    def dist_job_to_tasks(self) -> Dict[str, Tuple[str, ...]]:
        dist_job_to_tasks: Dict[str, Tuple[str, ...]] = dict()
        for g, job_to_task_assignments in self.GPU_to_task_assignments:
            for job_ID, task_assignments in job_to_task_assignments.items():
                if len(task_assignments) > 1:
                    dist_job_to_tasks[job_ID] = tuple(
                        sorted([assignment.task.task_ID for assignment in task_assignments]))
        return dist_job_to_tasks

    def task_comp_mem_requirements(self) -> Dict[str, Tuple[int, int]]:
        requirements: Dict[str, Tuple[int, int]] = dict()
        for g, job_to_task_assignments in self.GPU_to_task_assignments:
            for job_ID, task_assignments in job_to_task_assignments.items():
                for assignment in task_assignments:
                    assert isinstance(assignment, TaskAssignment)
                    requirements[assignment.task.task_ID] = (assignment.comp_proportion, assignment.memory)
        return requirements

    @staticmethod
    def merge_solver_assignments(*solver_assignments: Dict[str, Set[str]]):
        merged: Dict[str, Set[str]] = dict()
        for assignments in solver_assignments:
            for GPU_ID, task_IDs in assignments.items():
                assert GPU_ID not in merged
                merged[GPU_ID] = task_IDs
        return merged

    @classmethod
    def from_solver_assigment(cls,
                              cluster: Cluster,
                              GPU_type_to_task_comp_mem_requirements_and_profits: Dict[
                                  GPUType, Dict[str, Tuple[int, int, Union[int, float]]]],
                              solver_assignments: Dict[str, Set[str]]) -> 'Assignments':
        GPU_to_task_assignments: Dict[str, Dict[str, Set[TaskAssignment]]] = defaultdict(lambda: defaultdict(set))
        for GPU_ID, task_ID in solver_assignments:
            g = cluster.get_GPU_by_ID(GPU_ID)
            task_comp_mem_requirements_and_profits = GPU_type_to_task_comp_mem_requirements_and_profits[g.GPU_type]
            comp, mem, _ = task_comp_mem_requirements_and_profits[task_ID]
            job_ID = Task.task_ID_to_job_ID(task_ID)
            cluster.get_undone_job(job_ID)
            task_assignments = GPU_to_task_assignments[GPU_ID][job_ID]
            assert task_ID not in task_assignments
            task_assignments.add(task_ID)
        return Assignments(GPU_to_task_assignments=GPU_to_task_assignments)

    def job_ID_to_GPU_IDs(self) -> Dict[str, Set[str]]:
        d: Dict[str, Set[str]] = defaultdict(set)
        for job_ID, task_assignments in self.job_ID_to_task_assignments:
            for task_assignment in task_assignments:
                d[job_ID] = task_assignment.GPU_type
        return d

    @staticmethod
    def preemptive_overheads(data_source: DataSource, assignments_1: 'Assignments', assignments_2: 'Assignments'):
        assignments_1_job_ID_to_GPU_IDs = assignments_1.job_ID_to_GPU_IDs()
        assignments_2_job_ID_to_GPU_IDs = assignments_2.job_ID_to_GPU_IDs()
        intersected_job_IDs = set(assignments_1_job_ID_to_GPU_IDs.keys()).intersection(assignments_2_job_ID_to_GPU_IDs.keys())
        overheads: Dict[str, int] = dict()
        c = get_config()
        for job_ID in intersected_job_IDs:
            assignments_1_GPU_IDs = assignments_1_job_ID_to_GPU_IDs[job_ID]
            assignments_2_GPU_IDs = assignments_2_job_ID_to_GPU_IDs[job_ID]
            diff = assignments_2_GPU_IDs.symmetric_difference(assignments_1_GPU_IDs)
            if len(diff) > 0:
                job_spec = data_source.job_specs_dict[job_ID]
                overhead = int(1e9 * np.random.uniform(*c.model_configs[job_spec.model_name].preemptive_overhead))
                overheads[job_ID] = overhead
        return overheads

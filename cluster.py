from collections import defaultdict
from typing import Dict, List, DefaultDict, Optional, Set, Union, Tuple

import numpy as np

from config import ClusterConfig, get_config
from data_source import DataSource
from object import GPU, GPUType, Job, Task, CompCapacity
from profit import ProfitCalculator
from copy import deepcopy


class Cluster:
    def __init__(self, cluster_config: ClusterConfig):
        self.GPUs: DefaultDict[GPUType, List[GPU]] = defaultdict(list)
        self.GPU_IDs: List[str] = list()
        self.GPU_ID_to_GPU: Dict[str, GPU] = dict()
        self.GPU_ID_to_GPU_Type: Dict[str, GPUType] = dict()
        for GPU_type, count in cluster_config.GPUs:
            for i in range(count):
                g = GPU(GPU_type, i)
                self.GPUs[GPU_type].append(g)
                self.GPU_IDs.append(g.GPU_ID)
                self.GPU_ID_to_GPU[g.GPU_ID] = g
                self.GPU_ID_to_GPU_Type[g.GPU_ID] = g.GPU_type

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
        for GPU_type, job_ID_to_task_assignments in self.assignments.GPU_type_to_task_assignments:
            p = profit_calculator.calculate_jobs(data_source=data_source, job_IDs=job_ID_to_task_assignments.keys(),
                                                 GPU_type=GPU_type)
            total_profit += p
        return total_profit


class TaskAssignment:
    def __init__(self, GPU_ID: str, GPU_type: GPUType, task: Task, comp_proportion: int, memory: int, over_supplied: int=0):
        self.GPU_ID: str = GPU_ID
        self.GPU_type: GPUType = GPU_type
        self.task: Task = task
        self.comp_proportion: int = comp_proportion
        self.memory: int = memory
        self.over_supplied: int = over_supplied

    def __hash__(self):
        return hash(self.task)

    def __eq__(self, other):
        return self.task == other.task

    def __ne__(self, other):
        return self.task != other.task


class Assignments:
    def __init__(self, GPU_type_to_task_assignments: Optional[Dict[GPUType, Dict[str, Set[TaskAssignment]]]] = None):
        if GPU_type_to_task_assignments is None:
            GPU_type_to_task_assignments = dict()
        self.GPU_type_to_task_assignments: Dict[GPUType, Dict[str, Set[TaskAssignment]]] = GPU_type_to_task_assignments
        self.__init_views()

    def __init_views(self):
        self.job_ID_to_task_assignments: Dict[str, Set[TaskAssignment]] = dict()
        for GPU_type, job_ID_task_assignments in self.GPU_type_to_task_assignments.items():
            for job_ID, task_assignments in job_ID_task_assignments.items():
                self.job_ID_to_task_assignments[job_ID] = task_assignments

    def to_solver_assignments(self) -> Dict[str, Set[str]]:
        d: Dict[str, Set[str]] = defaultdict(set)
        for _, job_to_task_assignments in self.GPU_type_to_task_assignments.items():
            for job, task_assignments in job_to_task_assignments.items():
                for task_assignment in task_assignments:
                    d[task_assignment.GPU_ID].add(task_assignment.task.task_ID)
        return d

    def dist_job_to_tasks(self) -> Dict[str, Tuple[str, ...]]:
        dist_job_to_tasks: Dict[str, Tuple[str, ...]] = dict()
        for _, job_to_task_assignments in self.GPU_type_to_task_assignments.items():
            for job_ID, task_assignments in job_to_task_assignments.items():
                if len(task_assignments) > 1:
                    dist_job_to_tasks[job_ID] = tuple(
                        sorted([assignment.task.task_ID for assignment in task_assignments]))
        return dist_job_to_tasks

    def task_comp_mem_requirements(self) -> Dict[str, Tuple[int, int]]:
        requirements: Dict[str, Tuple[int, int]] = dict()
        for _, job_to_task_assignments in self.GPU_type_to_task_assignments:
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
                              GPU_ID_to_GPU_type: Dict[str, GPUType],
                              GPU_type_to_task_comp_mem_requirements: Dict[
                                  GPUType, Dict[str, Tuple[int, int]]],
                              solver_assignments: Dict[str, Set[str]]) -> 'Assignments':
        GPU_type_to_task_assignments: Dict[GPUType, Dict[str, Set[TaskAssignment]]] = defaultdict(lambda: defaultdict(set))
        for GPU_ID, task_IDs in solver_assignments.items():
            for task_ID in task_IDs:
                GPU_type = GPU_ID_to_GPU_type[GPU_ID]
                task_comp_mem_requirements = GPU_type_to_task_comp_mem_requirements[GPU_type]
                comp, mem = task_comp_mem_requirements[task_ID]
                job_ID = Task.task_ID_to_job_ID(task_ID)
                task_assignments = GPU_type_to_task_assignments[GPU_type][job_ID]
                assert task_ID not in task_assignments
                task_assignments.add(TaskAssignment(GPU_ID=GPU_ID, GPU_type=GPU_type, task=Task.from_task_ID(task_ID), comp_proportion=comp, memory=mem))
        return Assignments(GPU_type_to_task_assignments=GPU_type_to_task_assignments)

    def job_ID_to_GPU_IDs(self) -> Dict[str, Set[str]]:
        d: Dict[str, Set[str]] = defaultdict(set)
        for job_ID, task_assignments in self.job_ID_to_task_assignments.items():
            for task_assignment in task_assignments:
                d[job_ID].add(task_assignment.GPU_ID)
        return d

    @staticmethod
    def preemptive_overheads(data_source: DataSource, assignments_1: 'Assignments', assignments_2: 'Assignments'):
        assignments_1_job_ID_to_GPU_IDs = assignments_1.job_ID_to_GPU_IDs()
        assignments_2_job_ID_to_GPU_IDs = assignments_2.job_ID_to_GPU_IDs()
        intersected_job_IDs = set(assignments_1_job_ID_to_GPU_IDs.keys()).intersection(
            assignments_2_job_ID_to_GPU_IDs.keys())
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

    def supplement_over_supply(self) -> 'Assignments':
        dist_jobs: Set[str] = set()
        job_ID_to_GPU_IDs = self.job_ID_to_GPU_IDs()
        GPU_ID_to_remaining_comp: DefaultDict[str, int] = defaultdict(lambda :CompCapacity)
        GPU_ID_to_job_size: DefaultDict[str, int] = defaultdict(lambda :0)
        job_ID_to_task_assignments: Dict[str, Set[TaskAssignment]] = defaultdict(set)
        job_ID_to_GPU_type: Dict[str, GPUType] = dict()
        for GPU_type, job_to_task_assignments in self.GPU_type_to_task_assignments.items():
            for job_ID, task_assignments in job_to_task_assignments.items():
                job_ID_to_GPU_type[job_ID] = GPU_type
                if len(task_assignments) > 1:
                    dist_jobs.add(job_ID)
                job_ID_to_task_assignments[job_ID] = deepcopy(task_assignments)
                GPU_IDs = job_ID_to_GPU_IDs[job_ID]
                comp = next(iter(task_assignments)).comp_proportion
                for GPU_ID in GPU_IDs:
                    GPU_ID_to_remaining_comp[GPU_ID] -= comp
                    GPU_ID_to_job_size[GPU_ID] += 1
        def supply(inn_job_ID: str):
            task_assignments_ = job_ID_to_task_assignments[inn_job_ID]
            least_supply = np.inf
            for task_assignment in task_assignments_:
                remaining_comp = GPU_ID_to_remaining_comp[task_assignment.GPU_ID]
                job_size = GPU_ID_to_job_size[task_assignment.GPU_ID]
                supply_ = remaining_comp // job_size
                if supply_ < least_supply:
                    least_supply = supply_
            for task_assignment in task_assignments_:
                task_assignment.over_supplied = least_supply
                GPU_ID_to_remaining_comp[task_assignment.GPU_ID] -= least_supply

        def remain_supply(inn_job_ID: str):
            task_assignments_ = job_ID_to_task_assignments[inn_job_ID]
            least_supply = np.inf
            for task_assignment in task_assignments_:
                remaining_comp = GPU_ID_to_remaining_comp[task_assignment.GPU_ID]
                if remaining_comp < least_supply:
                    least_supply = remaining_comp
            for task_assignment in task_assignments_:
                task_assignment.over_supplied += least_supply
                GPU_ID_to_remaining_comp[task_assignment.GPU_ID] -= least_supply

        for dist_job in dist_jobs:
            supply(inn_job_ID=dist_job)
        for dist_job in dist_jobs:
            task_assignments = job_ID_to_task_assignments[dist_job]
            for task_assignment in task_assignments:
                GPU_ID_to_job_size[task_assignment.GPU_ID] -= 1
        for job_ID in job_ID_to_task_assignments:
            if job_ID not in dist_jobs:
                supply(inn_job_ID=job_ID)
        for job_ID in job_ID_to_task_assignments:
            if job_ID not in dist_jobs:
                remain_supply(inn_job_ID=job_ID)
        for job_ID in dist_jobs:
            remain_supply(inn_job_ID=job_ID)
        GPU_type_to_task_assignments: Dict[GPUType, Dict[str, Set[TaskAssignment]]] = defaultdict(dict)
        for job_ID, task_assignments in job_ID_to_task_assignments.items():
            GPU_type = job_ID_to_GPU_type[job_ID]
            GPU_type_to_task_assignments[GPU_type][job_ID] = task_assignments
        return Assignments(GPU_type_to_task_assignments=GPU_type_to_task_assignments)

    def jobs_iteration_throughput(self, data_source: DataSource) -> Dict[str, float]:
        d: Dict[str, float] = dict()
        for job_ID in self.job_ID_to_task_assignments:
            d[job_ID] = self.job_iteration_throughput(data_source=data_source, job_ID=job_ID)
        return d

    def job_iteration_throughput(self, data_source: DataSource, job_ID: str) -> float:
        task_assignments = self.job_ID_to_task_assignments[job_ID]
        assert len(task_assignments) > 0
        comp_proportion = {task_assignment.comp_proportion + task_assignment.over_supplied for task_assignment in task_assignments}
        assert len(comp_proportion) == 1
        comp_proportion = next(iter(comp_proportion))
        GPU_type = {task_assignment.GPU_type for task_assignment in task_assignments}
        assert len(GPU_type) == 1
        GPU_type = next(iter(GPU_type))
        worker_count = len(task_assignments)
        job_spec = data_source.get_job_spec(job_ID)
        return data_source.iteration_throughput(
            model_name=job_spec.model_name,
            batch_size=job_spec.batch_size,
            GPU_type=GPU_type,
            worker_count=worker_count,
            computation_proportion=comp_proportion
        )


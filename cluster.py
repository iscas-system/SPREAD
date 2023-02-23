from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, DefaultDict, Optional, Set, Tuple

import numpy as np

from config import ClusterConfig, get_config
from data_source import DataSource
from object import GPU, GPUType, Job, Task, TaskAssignment, CompCapacity, ProfitEnum
from profit import ProfitCalculator, get_profit_calculator
from itertools import count


class Cluster:
    def __init__(self, cluster_config: ClusterConfig):
        self.cluster_config: ClusterConfig = cluster_config
        self.assignments: Assignments = Assignments(cluster_config=self.cluster_config)
        self.jobs: Dict[str, Job] = dict()
        self.done_jobs: Dict[str, Job] = dict()

    def get_GPU_by_ID(self, GPU_ID: str) -> GPU:
        return self.cluster_config.GPU_ID_to_GPU[GPU_ID]

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
        self.done_jobs[job_ID] = self.jobs[job_ID]
        self.jobs.pop(job_ID)

    def ensure_start(self, now: int):
        for job_ID in self.assignments.job_ID_to_task_assignments:
            assert job_ID in self.jobs
            if self.jobs[job_ID].start_time is None:
                self.jobs[job_ID].start_time = now

    def add_preemptive_overhead(self, job_ID: str, overhead_iterations: float):
        self.jobs[job_ID].remaining_iterations += overhead_iterations

    def get_GPU_total_real_mem(self) -> int:
        total_real_mem = 0
        for GPU_ID in self.cluster_config.GPU_IDs:
            real_mem = GPUType.real_memory(GPU_type=self.cluster_config.get_GPU(GPU_ID).GPU_type)
            total_real_mem += real_mem
        return total_real_mem

    def running_status(self, data_source: DataSource) -> Dict:
        undone_jobs = self.jobs
        running_job_IDs, dist_jobs, spread_jobs = self.assignments.deployed_jobs(data_source)
        return {
            "running_jobs": len(running_job_IDs),
            "dist_jobs": len(dist_jobs),
            "spread_jobs": len(spread_jobs),
            "waiting_jobs": len(undone_jobs) - len(running_job_IDs),
            "done_jobs": len(self.done_jobs)
        }

class Assignments:
    def __init__(self, cluster_config: ClusterConfig, GPU_type_to_task_assignments: Optional[Dict[GPUType, Dict[str, Set[TaskAssignment]]]] = None):
        self.cluster_config: ClusterConfig = cluster_config
        if GPU_type_to_task_assignments is None:
            GPU_type_to_task_assignments = dict()
        self.GPU_type_to_task_assignments: Dict[GPUType, Dict[str, Set[TaskAssignment]]] = GPU_type_to_task_assignments
        self.__init_views()

    def __init_views(self):
        self.job_ID_to_task_assignments: Dict[str, Set[TaskAssignment]] = dict()
        for GPU_type, job_ID_task_assignments in self.GPU_type_to_task_assignments.items():
            for job_ID, task_assignments in job_ID_task_assignments.items():
                self.job_ID_to_task_assignments[job_ID] = task_assignments
        self.GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]] = self._get_GPU_ID_to_task_assignments()

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
        for job_to_task_assignments in self.GPU_type_to_task_assignments.values():
            for job_ID, task_assignments in job_to_task_assignments.items():
                for assignment in task_assignments:
                    assert isinstance(assignment, TaskAssignment)
                    requirements[assignment.task.task_ID] = (assignment.comp_req, assignment.memory)
        return requirements

    @staticmethod
    def merge_solver_assignments(*solver_assignments: Dict[str, Set[str]]):
        merged: Dict[str, Set[str]] = defaultdict(set)
        for assignments in solver_assignments:
            for GPU_ID, task_IDs in assignments.items():
                merged[GPU_ID] = merged[GPU_ID].union(task_IDs)
        return merged

    @classmethod
    def from_GPU_ID_to_task_assignments(cls,
                                        cluster_config: ClusterConfig,
                                        GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]]) -> 'Assignments':
        GPU_type_to_task_assignments: Dict[GPUType, Dict[str, Set[TaskAssignment]]] = defaultdict(
            lambda: defaultdict(set))
        for GPU_ID, task_assignments in GPU_ID_to_task_assignments.items():
            GPU_type = cluster_config.get_GPU(GPU_ID).GPU_type
            for task_assignment in task_assignments:
                job_ID = task_assignment.task.job_ID
                GPU_type_to_task_assignments[GPU_type][job_ID].add(task_assignment)
        return Assignments(cluster_config=cluster_config, GPU_type_to_task_assignments=GPU_type_to_task_assignments)

    @classmethod
    def from_job_ID_to_task_assignments(cls,
                                        cluster_config: ClusterConfig,
                                        job_ID_to_task_assignments: Dict[str, Set[TaskAssignment]]) -> 'Assignments':
        GPU_type_to_task_assignments: Dict[GPUType, Dict[str, Set[TaskAssignment]]] = defaultdict(
            lambda: defaultdict(set))
        for job_ID, task_assignments in job_ID_to_task_assignments.items():
            for task_assignment in task_assignments:
                GPU_type = task_assignment.GPU_type
                job_ID = task_assignment.task.job_ID
                GPU_type_to_task_assignments[GPU_type][job_ID].add(task_assignment)
        return Assignments(cluster_config=cluster_config, GPU_type_to_task_assignments=GPU_type_to_task_assignments)

    def job_ID_to_GPU_IDs(self) -> Dict[str, Set[str]]:
        d: Dict[str, Set[str]] = defaultdict(set)
        for job_ID, task_assignments in self.job_ID_to_task_assignments.items():
            for task_assignment in task_assignments:
                d[job_ID].add(task_assignment.GPU_ID)
        return d

    def _get_GPU_ID_to_task_assignments(self) -> Dict[str, Set[TaskAssignment]]:
        d: DefaultDict[str, Set[TaskAssignment]] = defaultdict(set)
        for task_assignments in self.job_ID_to_task_assignments.values():
            for task_assignment in task_assignments:
                d[task_assignment.GPU_ID].add(task_assignment)
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
        GPU_ID_to_remaining_comp: DefaultDict[str, int] = defaultdict(lambda: CompCapacity)
        GPU_ID_to_job_size: DefaultDict[str, int] = defaultdict(lambda: 0)
        job_ID_to_task_assignments: Dict[str, Set[TaskAssignment]] = defaultdict(set)
        job_ID_to_GPU_type: Dict[str, GPUType] = dict()
        for GPU_type, job_to_task_assignments in self.GPU_type_to_task_assignments.items():
            for job_ID, task_assignments in job_to_task_assignments.items():
                job_ID_to_GPU_type[job_ID] = GPU_type
                if len(task_assignments) > 1:
                    dist_jobs.add(job_ID)
                job_ID_to_task_assignments[job_ID] = deepcopy(task_assignments)
                GPU_IDs = job_ID_to_GPU_IDs[job_ID]
                comp = next(iter(task_assignments)).comp_req
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
                GPU_ID_to_job_size[task_assignment.GPU_ID] -= 1

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
                GPU_ID_to_job_size[task_assignment.GPU_ID] -= 1

        for dist_job in dist_jobs:
            supply(inn_job_ID=dist_job)
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
        return Assignments(cluster_config=self.cluster_config, GPU_type_to_task_assignments=GPU_type_to_task_assignments)

    def get_job_over_supply(self) -> Tuple[Dict[str, int], int]:
        job_over_supply: Dict[str, int] = dict()
        total_over_supply = 0
        for job_ID, task_assignments in self.job_ID_to_task_assignments.items():
            task_total_oversupply = 0
            for task_assignment in task_assignments:
                task_total_oversupply += task_assignment.over_supplied
                total_over_supply += task_assignment.over_supplied
            job_over_supply[job_ID] = task_total_oversupply
        return job_over_supply, total_over_supply

    def get_job_lack_supply(self, data_source: DataSource) -> Tuple[Dict[str, int], int]:
        job_lack_supply: Dict[str, int] = dict()
        total_lack_supply = 0
        for job_ID, task_assignments in self.job_ID_to_task_assignments.items():
            job_spec = data_source.get_job_spec(job_ID=job_ID)
            plan_total_comp = job_spec.plan_comp * job_spec.plan_worker_count
            task_total_comp = 0
            for task_assignment in task_assignments:
                task_total_comp += task_assignment.over_supplied + task_assignment.comp_req
            if task_total_comp >= plan_total_comp:
                continue
            lack_supply = plan_total_comp - task_total_comp
            total_lack_supply += lack_supply
            job_lack_supply[job_ID] = lack_supply
        return job_lack_supply, total_lack_supply

    def get_job_computation_utilization(self, data_source: DataSource) -> Tuple[Dict[str, float], float]:
        job_comp_util = dict()
        total_comp_util = 0
        for GPU_type, job_ID_to_task_assignments in self.GPU_type_to_task_assignments.items():
            for job_ID, task_assignments in job_ID_to_task_assignments.items():
                worker_count = len(task_assignments)
                task_assignment = next(iter(task_assignments))
                comp_req = task_assignment.over_supplied + task_assignment.comp_req
                task_comp_util = data_source.job_task_computation_utilization(job_ID=job_ID, GPU_type=GPU_type, worker_count=worker_count, comp_req=comp_req)
                job_comp_util[job_ID] = task_comp_util * worker_count
                total_comp_util += task_comp_util * worker_count
        return job_comp_util, total_comp_util

    def get_job_real_mem_utilization(self, data_source: DataSource) -> Tuple[Dict[str, float], float]:
        job_mem_util = dict()
        total_mem_util = 0
        for GPU_type, job_ID_to_task_assignments in self.GPU_type_to_task_assignments.items():
            for job_ID, task_assignments in job_ID_to_task_assignments.items():
                worker_count = len(task_assignments)
                task_original_mem, _ = data_source.get_job_task_memory(job_ID=job_ID, worker_count=worker_count)
                task_mem_util = task_original_mem
                job_mem_util[job_ID] = task_mem_util * worker_count
                total_mem_util += task_mem_util * worker_count
        return job_mem_util, total_mem_util

    def calc_profits(self, data_source: DataSource, profit_calculator: ProfitCalculator) -> float:
        total_profit = 0
        job_lack_supply, _ = self.get_job_lack_supply(data_source)
        for GPU_type, job_ID_to_task_assignments in self.GPU_type_to_task_assignments.items():
            p = profit_calculator.calculate_jobs(data_source=data_source,
                                                 job_IDs=set(job_ID_to_task_assignments.keys()),
                                                 cluster_config=self.cluster_config,
                                                 GPU_type=GPU_type,
                                                 job_lack_supply=job_lack_supply,
                                                 job_ID_to_task_assignments=job_ID_to_task_assignments)
            total_profit += np.sum(list(p.values()))
        return total_profit

    def jobs_iteration_time(self, data_source: DataSource) -> Dict[str, int]:
        d: Dict[str, int] = dict()
        for job_ID in self.job_ID_to_task_assignments:
            d[job_ID] = self.job_iteration_time_nano(data_source=data_source, job_ID=job_ID)
        return d

    def job_iteration_time_nano(self, data_source: DataSource, job_ID: str) -> int:
        task_assignments = self.job_ID_to_task_assignments[job_ID]
        assert len(task_assignments) > 0
        comp_req = {task_assignment.comp_req + task_assignment.over_supplied for task_assignment in
                           task_assignments}
        assert len(comp_req) == 1
        comp_req = next(iter(comp_req))
        GPU_type = {task_assignment.GPU_type for task_assignment in task_assignments}
        GPU_ID_to_node_id = self.cluster_config.GPU_ID_to_node_id

        cross_node = len({GPU_ID_to_node_id[task_assignment.GPU_ID] for task_assignment in task_assignments}) > 1
        assert len(GPU_type) == 1
        GPU_type = next(iter(GPU_type))
        worker_count = len(task_assignments)
        job_spec = data_source.get_job_spec(job_ID)
        return data_source.iteration_time_nano(
            model_name=job_spec.model_name,
            batch_size=job_spec.batch_size,
            GPU_type=GPU_type,
            worker_count=worker_count,
            cross_node=cross_node,
            comp_req=comp_req,
        )

    def deployed_jobs(self, data_source: DataSource) -> Tuple[Set[str], Set[str], Set[str]]:
        job_IDs = set(self.job_ID_to_task_assignments.keys())
        dist_jobs = set()
        spread_jobs = set()
        for job_ID, task_assignments in self.job_ID_to_task_assignments.items():
            worker_count = len(task_assignments)
            if worker_count > 1:
                dist_jobs.add(job_ID)
            job_spec = data_source.get_job_spec(job_ID)
            if worker_count > job_spec.plan_worker_count:
                spread_jobs.add(job_ID)
        return job_IDs, dist_jobs, spread_jobs

    def remove_jobs(self, job_IDs: Set[str]) -> 'Assignments':
        job_ID_to_task_assignments = deepcopy(self.job_ID_to_task_assignments)
        for job_ID in job_IDs:
            job_ID_to_task_assignments.pop(job_ID)
        return Assignments.from_job_ID_to_task_assignments(cluster_config=self.cluster_config, job_ID_to_task_assignments=job_ID_to_task_assignments)

    def merge(self, other: 'Assignments') -> 'Assignments':
        GPU_type_to_task_assignments: Dict[GPUType, Dict[str, Set[TaskAssignment]]] = defaultdict(lambda :defaultdict(set))
        def add(assignments: Dict[GPUType, Dict[str, Set[TaskAssignment]]]):
            for GPU_type, job_task_assignments in assignments.items():
                for job_ID, task_assignments in job_task_assignments.items():
                    GPU_type_to_task_assignments[GPU_type][job_ID] = {TaskAssignment(
                        GPU_ID=task_assignment.GPU_ID,
                        GPU_type=task_assignment.GPU_type,
                        task=Task(job_ID=task_assignment.task.job_ID, task_idx=task_assignment.task.task_idx),
                        comp_req=task_assignment.comp_req,
                        memory=task_assignment.memory) for task_assignment in task_assignments}
        add(self.GPU_type_to_task_assignments)
        add(other.GPU_type_to_task_assignments)
        return Assignments(cluster_config=self.cluster_config, GPU_type_to_task_assignments=GPU_type_to_task_assignments)

    def clone(self) -> 'Assignments':
        return self.merge(Assignments(cluster_config=self.cluster_config))

    def statistics(self, preemptive: bool, now: int, cluster: Cluster, data_source: DataSource) -> Dict:
        cluster_real_total_mem = cluster.get_GPU_total_real_mem()
        profit = self.calc_profits(data_source=data_source, profit_calculator=get_profit_calculator(ProfitEnum.Throughput))
        deployed_jobs, deployed_dist_jobs, deployed_spread_jobs = self.deployed_jobs(data_source)

        job_ID_assignment_repr: DefaultDict = defaultdict(list)
        for job_ID, task_assignments in self.job_ID_to_task_assignments.items():
            worker_count = len(task_assignments)
            comp = next(iter(task_assignments)).comp_req
            node_ids = {cluster.cluster_config.GPU_ID_to_node_id[t.GPU_ID] for t in task_assignments}
            cross_node = len(node_ids) > 1
            job_ID_assignment_repr[job_ID] = f"{job_ID}|{worker_count}|{comp}|{cross_node}"

        d = {
            "now": now,
            "preemptive": preemptive,
            "cluster_real_total_mem": cluster_real_total_mem,
            "profit": float(profit),
            "deployed_job_size": len(deployed_jobs),
            "deployed_dist_job_size": len(deployed_dist_jobs),
            "deployed_spread_job_size": len(deployed_spread_jobs),
            "job_ID_to_assignment_repr": job_ID_assignment_repr,
        }
        return d



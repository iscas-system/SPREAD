import time
from collections import defaultdict
from typing import Tuple, Optional, List, Dict, Set, Any

import numpy as np

from cluster import TaskAssignment, Assignments
from object import Job, GPUType, CompCapacity, Task
from scheduler import Scheduler
from .solver import SolverParameters, do_solve
from profit import get_profit_calculator


class MMKPScheduler(Scheduler):
    class ScoreWeights:
        def __init__(self, effective_resource_score_range: Tuple[float, float], balance_weight: float,
                     resource_weight: float):
            self.effective_resource_score_range: Tuple[float, float] = effective_resource_score_range
            self.balance_weight: float = balance_weight
            self.resource_weight: float = resource_weight

    def _init_config(self):
        self.throughput_degrade_threshold = 0.05
        self.GPU_type = GPUType.RTX_2080Ti
        self.plan_comp_unit = 1
        self.score_weights = [
            MMKPScheduler.ScoreWeights((0, 1.4), 1.5, 1.),
            MMKPScheduler.ScoreWeights((1.4, np.inf), 0.5, 1.5)
        ]
        self.resource_score_upper_bound = 1.6
        self.saturate_factor = 2
        self.saturate_partitions = 5
        ...

    class AssignmentPlan:
        def __init__(self,
                     job_ID: str,
                     worker_count: int,
                     comp_req: int,
                     mem_req: int,
                     GPU_type: GPUType,
                     score_weights: List['MMKPScheduler.ScoreWeights'],
                     direct_plan: Optional['MMKPScheduler.AssignmentPlan'],
                     ):
            self.job_ID: str = job_ID
            self.worker_count: int = worker_count
            self.comp_req: int = comp_req
            self.mem_req: int = mem_req
            self.GPU_type: GPUType = GPU_type
            self.score_weights: List['MMKPScheduler.ScoreWeights'] = score_weights
            self.direct_plan: Optional['MMKPScheduler.AssignmentPlan'] = direct_plan
            self.total_normalized_resource_req = self._total_normalized_resource_req()
            self.balance_score = self._balance_score()
            self.resource_score = self._resource_score(direct_plan)
            self.comprehensive_score = self._comprehensive_score()

        def _total_normalized_resource_req(self) -> float:
            total_normalized_resource_req = (self.comp_req / CompCapacity) * self.worker_count + \
                                            (self.mem_req / GPUType.normalized_memory(
                                                self.GPU_type)) * self.worker_count
            return total_normalized_resource_req

        def _balance_score(self) -> float:
            return abs(self.comp_req / CompCapacity - self.mem_req / GPUType.normalized_memory(self.GPU_type))

        def _resource_score(self, direct: 'MMKPScheduler.AssignmentPlan') -> float:
            if self.worker_count == 1:
                return 1
            splitting_total_normalized_resource_req = self.total_normalized_resource_req
            direct_total_normalized_resource_req = direct.total_normalized_resource_req
            return splitting_total_normalized_resource_req / direct_total_normalized_resource_req

        def _comprehensive_score(self) -> float:
            resource_score = self.resource_score
            balance_score = self.balance_score
            for score_weight in self.score_weights:
                r = score_weight.effective_resource_score_range
                if r[0] < resource_score < r[1]:
                    score = score_weight.balance_weight * balance_score + score_weight.balance_weight * resource_score
                    return score
            assert False

    def do_assign(self, preemptive: bool) -> Tuple[Assignments, Optional[Any]]:
        GPU_size = len(self.cluster.GPU_IDs)
        total_comp = GPU_size * CompCapacity
        GPU_mem = GPUType.normalized_memory(self.GPU_type)
        total_mem = GPU_mem * GPU_size
        total_normalized_comp = total_comp / CompCapacity
        total_normalized_mem = total_mem / GPU_mem
        total_normalized_resource = total_normalized_comp + total_normalized_mem
        all_job_IDs = set(self.cluster.jobs.keys())
        if not preemptive:
            GPU_ID_to_task_assignments = self.cluster.assignments.GPU_ID_to_task_assignments()
            all_job_IDs -= set(self.cluster.assignments.job_ID_to_task_assignments.keys())
        else:
            GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]] = defaultdict(set)
        all_job_IDs = sorted(list(all_job_IDs))
        GPU_comp_mem_capacity: Dict[str, Tuple[int, int]] = dict()
        for GPU_ID in self.cluster.GPU_IDs:
            GPU_comp_mem_capacity[GPU_ID] = (CompCapacity, GPU_mem)

        for GPU_ID, task_assignments in GPU_ID_to_task_assignments.items():
            for task_assignment in task_assignments:
                total_comp -= task_assignment.comp_req
                total_mem -= task_assignment.memory
                comp, mem = GPU_comp_mem_capacity[GPU_ID]
                GPU_comp_mem_capacity[GPU_ID] = comp - task_assignment.comp_req, mem - task_assignment.memory

        profit_calculator = get_profit_calculator(profit_enum=self.profit_enum)

        job_ID_to_profit: Dict[str, float] = profit_calculator.calculate_jobs(data_source=self.data_source, job_IDs=all_job_IDs, GPU_type=self.GPU_type)

        direct_assignment_plans: Dict[str, 'MMKPScheduler.AssignmentPlan'] = dict()
        split_assignment_plans: Dict[str, 'MMKPScheduler.AssignmentPlan'] = dict()
        for job_ID in all_job_IDs:
            direct_plan = self.job_direct_assignment_plan(job_ID=job_ID)
            direct_assignment_plans[job_ID] = direct_plan
            split_assignment_plans[job_ID] = self.job_splitting_assignment_plan(job_ID=job_ID, direct_plan=direct_plan)
        sorted_job_IDs = sorted(all_job_IDs, key=lambda j_ID: direct_assignment_plans[j_ID].balance_score)
        saturate_job_IDs: List[str] = list()

        total_consumed_comp = 0
        total_consumed_mem = 0
        for job_ID in sorted_job_IDs:
            assignment_plan = direct_assignment_plans[job_ID]
            total_consumed_comp += assignment_plan.comp_req
            total_consumed_mem += assignment_plan.mem_req
            total_normalized_consumed_resource = total_consumed_mem / GPU_mem + total_consumed_comp / CompCapacity
            if total_normalized_consumed_resource > total_normalized_resource * self.saturate_factor:
                break
            saturate_job_IDs.append(job_ID)

        splittable_saturate_job_IDs = list()
        in_splittable_saturate_job_IDs = list()
        for saturate_job_ID in saturate_job_IDs:
            if split_assignment_plans[saturate_job_ID] is not None or \
                    split_assignment_plans[saturate_job_ID].resource_score > self.resource_score_upper_bound:
                in_splittable_saturate_job_IDs.append(saturate_job_ID)
            else:
                splittable_saturate_job_IDs.append(saturate_job_ID)
        sorted_splittable_saturate_job_IDs = sorted(splittable_saturate_job_IDs,
                                         key=lambda j_ID: split_assignment_plans[j_ID].comprehensive_score)
        partition_count_to_solver_result = dict()
        partition_count_to_splitting_jobs = dict()
        for partition_count in range(self.saturate_partitions + 1):
            total_splittable_job_size = len(sorted_splittable_saturate_job_IDs)
            partition_size = total_splittable_job_size // self.saturate_partitions
            splitting_border = min(partition_size * partition_count, total_splittable_job_size)
            splitting_job_IDs = sorted_splittable_saturate_job_IDs[:splitting_border]
            direct_job_IDs = in_splittable_saturate_job_IDs[:]
            assignment_plans = \
            [split_assignment_plans[job_ID] for job_ID in splitting_job_IDs] +\
            [direct_assignment_plans[job_ID] for job_ID in direct_job_IDs]
            partition_count_to_splitting_jobs[partition_count] = splitting_job_IDs
            partial_job_ID_to_profit = {assignment_plan.job_ID: job_ID_to_profit[assignment_plan.job_ID] for assignment_plan in assignment_plans}
            solver_parameters = self.build_solver_parameters(
                assignment_plans=assignment_plans,
                job_ID_to_profit=partial_job_ID_to_profit,
                GPU_comp_mem_capacity=GPU_comp_mem_capacity)
            solver_result = do_solve(solver_params=solver_parameters)
            partition_count_to_solver_result[partition_count] = solver_result
        max_profit = 0
        max_profit_partition_count = None
        optimum_solver_result = None
        solver_durations = list()

        for pc, solver_result in partition_count_to_solver_result.items():
            solver_durations.append(solver_result.duration)
            if solver_result.profit > max_profit:
                max_profit = solver_result.profit
                optimum_solver_result = solver_result
                max_profit_partition_count = pc

        optimum_assignment = optimum_solver_result.assignment
        if not preemptive:
            optimum_assignment = Assignments.merge_solver_assignments(self.cluster.assignments.to_solver_assignments(), optimum_solver_result.assignment)
        GPU_type_to_task_comp_mem_requirements: Dict[
            GPUType, Dict[str, Tuple[int, int]]] = {self.GPU_type: dict()}
        for task_ID, data in optimum_solver_result.solver_parameters.task_comp_mem_requirements_and_profits.items():
            GPU_type_to_task_comp_mem_requirements[self.GPU_type][task_ID] = (data[0], data[1])
        assignments = Assignments.from_solver_assigment(GPU_ID_to_GPU_type=self.cluster.GPU_ID_to_GPU_type,
                                          GPU_type_to_task_comp_mem_requirements=GPU_type_to_task_comp_mem_requirements,
                                          solver_assignments=optimum_assignment)
        splitting_job_IDs = partition_count_to_splitting_jobs[max_profit_partition_count]
        total_splitting_job_supplied_comp = 0
        total_splitting_job_supplied_mem = 0
        job_ID_to_supplied_comp = dict()
        job_ID_to_supplied_mem = dict()
        for splitting_job_ID in splitting_job_IDs:
            splitting_assignment_plan = split_assignment_plans[splitting_job_ID]
            direct_assignment_plan = direct_assignment_plans[splitting_job_ID]
            supplied_comp = splitting_assignment_plan.comp_req - direct_assignment_plan.comp_req
            total_splitting_job_supplied_comp += supplied_comp
            job_ID_to_supplied_comp[splitting_job_ID] = supplied_comp
            supplied_mem = splitting_assignment_plan.mem_req * splitting_assignment_plan.worker_count - direct_assignment_plan.mem_req * direct_assignment_plan.worker_count
            total_splitting_job_supplied_mem += supplied_mem
            job_ID_to_supplied_mem[splitting_job_ID] = supplied_mem

        statistics = {
            "solver_durations": solver_durations,
            "splitting_job_size": splitting_job_IDs,
            "total_splitting_job_supplied_comp": total_splitting_job_supplied_comp,
            "total_splitting_job_supplied_mem": total_splitting_job_supplied_mem,
            "job_supplied_comp": job_ID_to_supplied_comp,
            "job_supplied_mem": job_ID_to_supplied_mem
        }
        return assignments, statistics

    def build_solver_parameters(self,
                                assignment_plans: List['MMKPScheduler.AssignmentPlan'],
                                job_ID_to_profit: Dict[str, float],
                                GPU_comp_mem_capacity: Dict[str, Tuple[int, int]]) -> SolverParameters:
        #     solver_type: SolverEnum
        #     GPU_type: GPUType
        #     dist_job_to_tasks: Dict[str, Tuple[str, ...]]
        #     GPU_comp_mem_capacity: Dict[str, Tuple[int, int]]
        #     task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, Union[int, float]]]
        task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, float]] = dict()
        dist_job_to_tasks: Dict[str, Tuple[str, ...]] = dict()
        for assignment_plan in assignment_plans:
            job_ID = assignment_plan.job_ID
            task_IDs = list()
            for i in range(assignment_plan.worker_count):
                task = Task(job_ID=job_ID, task_idx=i)
                task_IDs.append(task.task_ID)
                task_comp_mem_requirements_and_profits[task.task_ID] = (assignment_plan.comp_req, assignment_plan.mem_req, job_ID_to_profit[job_ID])
            if len(task_IDs) > 1:
                dist_job_to_tasks[job_ID] = tuple(task_IDs)

        solver_parameters = SolverParameters(
            solver_type=self.solver_enum,
            GPU_type=self.GPU_type,
            dist_job_to_tasks=dist_job_to_tasks,
            GPU_comp_mem_capacity=GPU_comp_mem_capacity,
            task_comp_mem_requirements_and_profits=task_comp_mem_requirements_and_profits
        )
        return solver_parameters

    def job_direct_assignment_plan(self, job_ID: str) -> 'MMKPScheduler.AssignmentPlan':
        job_spec = self.data_source.get_job_spec(job_ID=job_ID)
        mem_requirement = self.data_source.get_job_mem_requirement(
            job_ID=job_ID,
            GPU_type=self.GPU_type,
            worker_count=job_spec.plan_worker_count)
        return MMKPScheduler.AssignmentPlan(
            job_ID=job_ID,
            worker_count=job_spec.plan_worker_count,
            comp_req=job_spec.plan_comp,
            mem_req=mem_requirement,
            GPU_type=self.GPU_type,
        score_weights=self.score_weights,
        direct_plan=None)

    def job_splitting_assignment_plan(self, job_ID: str, direct_plan: 'MMKPScheduler.AssignmentPlan') -> Optional['MMKPScheduler.AssignmentPlan']:
        job_spec = self.data_source.get_job_spec(job_ID=job_ID)
        original_iteration_time = self.data_source.job_iteration_time(job_ID=job_ID, GPU_type=self.GPU_type,
                                                                      comp_req=job_spec.plan_comp,
                                                                      worker_count=job_spec.plan_worker_count)
        split_plan_worker_count = 2 * job_spec.plan_worker_count
        if job_spec.plan_comp % 2 == 0:
            split_plan_comp = job_spec.plan_comp // 2
        else:
            split_plan_comp = job_spec.plan_comp // 2 + 1
        no_suitable_splitting = False
        while True:
            iteration_time = self.data_source.job_iteration_time(
                job_ID=job_ID,
                GPU_type=self.GPU_type,
                comp_req=split_plan_comp,
                worker_count=split_plan_worker_count)
            if split_plan_comp > CompCapacity:
                no_suitable_splitting = True
                break
            if iteration_time < original_iteration_time:
                break
            if (iteration_time - original_iteration_time) / original_iteration_time < self.throughput_degrade_threshold:
                break
            split_plan_comp += self.plan_comp_unit
        if no_suitable_splitting:
            return None
        mem_requirement = self.data_source.get_job_mem_requirement(
            job_ID=job_ID,
            GPU_type=self.GPU_type,
            worker_count=split_plan_worker_count)
        return MMKPScheduler.AssignmentPlan(job_ID,
                                            split_plan_worker_count,
                                            split_plan_comp,
                                            mem_requirement,
                                            self.GPU_type,
                                            self.score_weights,
                                            direct_plan)

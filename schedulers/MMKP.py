from collections import defaultdict
from copy import deepcopy
from typing import Tuple, Optional, List, Dict, Set, Any, Callable

import numpy as np

from cluster import TaskAssignment, Assignments
from log import info
from object import GPUType, CompCapacity, Task
from profit import get_profit_calculator
from scheduler import Scheduler
from .solver import SolverParameters, do_solve, SolverResult
from .RR import RRScheduler


class MMKPScheduler(Scheduler):
    class ScoreWeights:
        def __init__(self, effective_resource_score_range: Tuple[float, float], balance_weight: float,
                     resource_weight: float):
            self.effective_resource_score_range: Tuple[float, float] = effective_resource_score_range
            self.balance_weight: float = balance_weight
            self.resource_weight: float = resource_weight

    def _init_config(self):
        self.throughput_degrade_threshold = self.config.get("throughput_degrade_threshold", 0.05)
        self.GPU_type = GPUType.RTX_2080Ti
        self.plan_comp_unit = self.config.get("plan_comp_unit", 1)
        self.score_weights = [
            MMKPScheduler.ScoreWeights((0, 0.2), 4, 1),
            MMKPScheduler.ScoreWeights((0.2, np.inf), 4, 8)
        ]
        self.resource_score_upper_bound = self.config.get("resource_score_upper_bound", 0.5)
        self.saturate_factor = self.config.get("saturate_factor", 4)
        self.non_preemptive_saturate_factor = self.config.get("non_preemptive_saturate_factor", 64)
        self.saturate_partitions = self.config.get("saturate_partitions", 10)
        self.trail_best_splittable_max_depth = self.config.get("trail_best_splittable_max_depth", 10)
        self.exploration_max_depth = self.config.get("exploration_max_depth", 20)
        self.use_round_robin_resource_ratio = self.config.get("use_round_robin_resource_ratio", 0.75)
        self.solver_duration_upper_bound = self.config.get("solver_duration_upper_bound", 30)

    class AssignmentPlan:
        def __init__(self,
                     job_ID: str,
                     worker_count: int,
                     comp_req: int,
                     mem_req: int,
                     GPU_type: GPUType,
                     score_weights: List['MMKPScheduler.ScoreWeights'],
                     direct_plan: Optional['MMKPScheduler.AssignmentPlan'],
                     iteration_time: float,
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
            self.iteration_time: float = iteration_time
            # for splitting assignment plans
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
            if self.direct_plan is None:
                return 1
            splitting_total_normalized_resource_req = self.total_normalized_resource_req
            direct_total_normalized_resource_req = direct.total_normalized_resource_req
            return splitting_total_normalized_resource_req / direct_total_normalized_resource_req - 1

        def _comprehensive_score(self) -> float:
            if self.direct_plan is None:
                return 0.
            resource_score = self.resource_score
            balance_score = self.balance_score
            balance_score_increment = balance_score - self.direct_plan.balance_score
            for score_weight in self.score_weights:
                r = score_weight.effective_resource_score_range
                if r[0] < resource_score < r[1]:
                    score = score_weight.balance_weight * balance_score_increment + score_weight.resource_weight * resource_score
                    return score
            assert False

    def use_round_robin(self, all_job_IDs: Set[str], total_normalized_resource: float, preemptive: bool) -> Optional[Tuple[Assignments, Optional[Any]]]:
        total_plan_comp = 0
        total_plan_mem = 0
        for job_ID in all_job_IDs:
            job_spec = self.data_source.get_job_spec(job_ID)
            total_plan_comp += job_spec.plan_comp * job_spec.plan_worker_count
            _, task_mem = self.data_source.get_job_task_memory(job_ID=job_ID,
                                                 worker_count=job_spec.plan_worker_count)
            plan_mem = task_mem * job_spec.plan_worker_count
            total_plan_mem += plan_mem
        total_consumed_normalized_comp = total_plan_comp / CompCapacity
        total_consumed_normalized_mem = total_plan_mem / GPUType.normalized_memory(self.GPU_type)
        total_consumed_normalized_resource = total_consumed_normalized_comp + total_consumed_normalized_mem
        if total_consumed_normalized_resource < total_normalized_resource * self.use_round_robin_resource_ratio:
            info("MMKP: Jobs are not enough to saturate, use RR instead.")
            sche = RRScheduler(self.name, self.scheduler_enum, None, None, self.data_source, self.cluster, {})
            return sche.do_assign(preemptive)
        return None

    def do_assign(self, preemptive: bool) -> Tuple[Assignments, Optional[Any]]:
        GPU_size = len(self.cluster.GPU_IDs)
        total_comp = GPU_size * CompCapacity
        GPU_mem = GPUType.normalized_memory(self.GPU_type)
        total_mem = GPU_mem * GPU_size
        total_normalized_comp = total_comp / CompCapacity
        total_normalized_mem = total_mem / GPU_mem
        all_job_IDs = set(self.cluster.jobs.keys())
        if not preemptive:
            GPU_ID_to_task_assignments = self.cluster.assignments.GPU_ID_to_task_assignments
            all_job_IDs -= set(self.cluster.assignments.job_ID_to_task_assignments.keys())
        else:
            GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]] = defaultdict(set)
        all_job_IDs = sorted(list(all_job_IDs))
        GPU_comp_mem_capacity: Dict[str, Tuple[int, int]] = dict()
        for GPU_ID in self.cluster.GPU_IDs:
            GPU_comp_mem_capacity[GPU_ID] = (CompCapacity, GPU_mem)

        total_normalized_consumed_comp = 0
        total_normalized_consumed_mem = 0
        if not preemptive:
            for GPU_ID, task_assignments in GPU_ID_to_task_assignments.items():
                for task_assignment in task_assignments:
                    total_normalized_consumed_comp += task_assignment.comp_req / CompCapacity
                    total_normalized_consumed_mem += task_assignment.memory / GPU_mem
                    comp, mem = GPU_comp_mem_capacity[GPU_ID]
                    GPU_comp_mem_capacity[GPU_ID] = comp - task_assignment.comp_req, mem - task_assignment.memory
        total_remain_normalized_resource = total_normalized_comp - total_normalized_consumed_comp + total_normalized_mem - total_normalized_consumed_mem
        info(f"MMKP starts do assign, preemptive: {preemptive}, total_normalized_comp: {total_normalized_comp}, "
             f"total_normalized_mem: {total_normalized_mem}, total_normalized_consumed_comp: {total_normalized_consumed_comp},"
             f"total_normalized_consumed_mem: {total_normalized_consumed_mem}, total_remain_normalized_resource: {total_remain_normalized_resource}")
        use_round_robin_result = self.use_round_robin(all_job_IDs=set(all_job_IDs), total_normalized_resource=total_remain_normalized_resource, preemptive=preemptive)
        if use_round_robin_result is not None:
            return use_round_robin_result

        profit_calculator = get_profit_calculator(profit_enum=self.profit_enum)

        job_ID_to_profit: Dict[str, float] = profit_calculator.calculate_jobs(data_source=self.data_source,
                                                                              job_IDs=all_job_IDs,
                                                                              GPU_type=self.GPU_type)

        direct_assignment_plans: Dict[str, 'MMKPScheduler.AssignmentPlan'] = dict()
        split_assignment_plans: Dict[str, 'MMKPScheduler.AssignmentPlan'] = dict()
        for job_ID in all_job_IDs:
            direct_plan = self.job_direct_assignment_plan(job_ID=job_ID)
            direct_assignment_plans[job_ID] = direct_plan
            split_assignment_plans[job_ID] = self.job_splitting_assignment_plan(job_ID=job_ID, direct_plan=direct_plan)

        selectors: List[Callable] = [self.select_saturate_jobs_by_balancing_total_comp_mem]
        optimum_solver_result, splitting_job_IDs = None, None
        solver_durations = list()
        for selector in selectors:
            saturate_job_IDs = selector(
                preemptive=preemptive,
                total_normalized_consumed_comp=total_normalized_consumed_comp,
                total_normalized_consumed_mem=total_normalized_consumed_mem,
                total_normalized_comp=total_normalized_comp,
                total_normalized_mem=total_normalized_mem,
                direct_assignment_plans=direct_assignment_plans,
                split_assignment_plans=split_assignment_plans,
                all_job_IDs=all_job_IDs)
            if len(saturate_job_IDs) == 0:
                continue
            sorted_splittable_saturate_job_IDs, in_splittable_saturate_job_IDs = self.extract_splittable_jobs_from_saturate_job_IDs(
                split_assignment_plans, saturate_job_IDs)
            optimum_solver_result_, splitting_job_IDs_, solver_durations_ = self.solve_saturate_job_IDs_by_partitions(
                job_ID_to_profit,
                direct_assignment_plans,
                split_assignment_plans,
                GPU_comp_mem_capacity,
                sorted_splittable_saturate_job_IDs,
                in_splittable_saturate_job_IDs)
            solver_durations.extend(solver_durations_)
            if optimum_solver_result is None or optimum_solver_result_.profit > optimum_solver_result.profit:
                optimum_solver_result = optimum_solver_result_
                splitting_job_IDs = splitting_job_IDs_

        if optimum_solver_result is None:
            return self.cluster.assignments.clone(), MMKPScheduler.build_statistics()
        optimum_assignment = optimum_solver_result.assignment
        GPU_type_to_task_comp_mem_requirements: Dict[
            GPUType, Dict[str, Tuple[int, int]]] = {self.GPU_type: dict()}
        for task_ID, data in optimum_solver_result.solver_parameters.task_comp_mem_requirements_and_profits.items():
            GPU_type_to_task_comp_mem_requirements[self.GPU_type][task_ID] = (data[0], data[1])
        assignments = Assignments.from_solver_assigment(GPU_ID_to_GPU_type=self.cluster.GPU_ID_to_GPU_type,
                                                        GPU_type_to_task_comp_mem_requirements=GPU_type_to_task_comp_mem_requirements,
                                                        solver_assignments=optimum_assignment)
        if not preemptive:
            assignments = self.cluster.assignments.merge(assignments)
        total_splitting_job_supplied_comp = 0
        total_splitting_job_supplied_mem = 0
        job_ID_to_supplied_comp = dict()
        job_ID_to_supplied_mem = dict()
        for splitting_job_ID in splitting_job_IDs:
            if splitting_job_ID not in assignments.job_ID_to_task_assignments:
                continue
            splitting_assignment_plan = split_assignment_plans[splitting_job_ID]
            direct_assignment_plan = direct_assignment_plans[splitting_job_ID]
            supplied_comp = splitting_assignment_plan.worker_count * splitting_assignment_plan.comp_req - direct_assignment_plan.comp_req * direct_assignment_plan.worker_count
            total_splitting_job_supplied_comp += supplied_comp
            job_ID_to_supplied_comp[splitting_job_ID] = supplied_comp
            supplied_mem = splitting_assignment_plan.mem_req * splitting_assignment_plan.worker_count - direct_assignment_plan.mem_req * direct_assignment_plan.worker_count
            total_splitting_job_supplied_mem += supplied_mem
            job_ID_to_supplied_mem[splitting_job_ID] = supplied_mem
        assignments = assignments.supplement_over_supply()
        statistics = MMKPScheduler.build_statistics(solver_durations=solver_durations,
                                                    splitting_job_IDs=splitting_job_IDs,
                                                    total_splitting_job_supplied_comp=total_splitting_job_supplied_comp,
                                                    total_splitting_job_supplied_mem=total_splitting_job_supplied_mem,
                                                    job_ID_to_supplied_comp=job_ID_to_supplied_comp,
                                                    job_ID_to_supplied_mem=job_ID_to_supplied_mem)
        return assignments, statistics

    def solve_assignment_plans(self,
                               assignment_plans_: List['MMKPScheduler.AssignmentPlan'],
                               job_ID_to_profit: Dict[str, float],
                               GPU_comp_mem_capacity: Dict[str, Tuple[int, int]], ):
        partial_job_ID_to_profit_ = {assignment_plan.job_ID: job_ID_to_profit[assignment_plan.job_ID] for
                                     assignment_plan in assignment_plans_}
        solver_parameters_ = self.build_solver_parameters(
            assignment_plans=assignment_plans_,
            job_ID_to_profit=partial_job_ID_to_profit_,
            GPU_comp_mem_capacity=GPU_comp_mem_capacity)
        return do_solve(solver_params=solver_parameters_)

    def solve_saturate_job_IDs_by_partitions(self,
                                             job_ID_to_profit: Dict[str, float],
                                             direct_assignment_plans: Dict[str, 'MMKPScheduler.AssignmentPlan'],
                                             split_assignment_plans: Dict[str, 'MMKPScheduler.AssignmentPlan'],
                                             GPU_comp_mem_capacity: Dict[str, Tuple[int, int]],
                                             sorted_splittable_saturate_job_IDs_: List[str],
                                             in_splittable_saturate_job_IDs_: List[str]):
        trail_count_to_solver_result = dict()
        trail_count_to_splitting_jobs = dict()
        total_splittable_job_size = len(sorted_splittable_saturate_job_IDs_)
        partition_size = total_splittable_job_size // self.saturate_partitions
        if partition_size == 0:
            partition_size = 1
        accumulated_partition_size = 0
        trail_count = 0

        if total_splittable_job_size == 0:
            assignment_plans = \
                [direct_assignment_plans[j_ID] for j_ID in in_splittable_saturate_job_IDs_[:]]
            trail_count_to_splitting_jobs[trail_count] = list()
            solver_result = self.solve_assignment_plans(
                assignment_plans_=assignment_plans,
                job_ID_to_profit=job_ID_to_profit,
                GPU_comp_mem_capacity=GPU_comp_mem_capacity)
            trail_count_to_solver_result[trail_count] = solver_result
        while accumulated_partition_size < total_splittable_job_size:
            splitting_border = min(partition_size * trail_count, total_splittable_job_size)
            splitting_job_IDs__ = sorted_splittable_saturate_job_IDs_[:splitting_border]
            direct_job_IDs = sorted_splittable_saturate_job_IDs_[splitting_border:] + in_splittable_saturate_job_IDs_[:]
            assignment_plans = \
                [split_assignment_plans[j_ID] for j_ID in splitting_job_IDs__] + \
                [direct_assignment_plans[j_ID] for j_ID in direct_job_IDs]
            trail_count_to_splitting_jobs[trail_count] = splitting_job_IDs__
            solver_result = self.solve_assignment_plans(
                assignment_plans_=assignment_plans,
                job_ID_to_profit=job_ID_to_profit,
                GPU_comp_mem_capacity=GPU_comp_mem_capacity
            )
            if solver_result.duration / 1e9 > self.solver_duration_upper_bound:
                break
            trail_count_to_solver_result[trail_count] = solver_result
            trail_count += 1
            accumulated_partition_size += partition_size
        max_profit = None
        max_profit_trail_count = None
        optimum_solver_result__ = None
        solver_durations__ = list()

        for tc, solver_result in trail_count_to_solver_result.items():
            solver_durations__.append(solver_result.duration)
            if max_profit is None or solver_result.profit >= max_profit:
                max_profit = solver_result.profit
                optimum_solver_result__ = solver_result
                max_profit_trail_count = tc
        return optimum_solver_result__, trail_count_to_splitting_jobs[
            max_profit_trail_count], solver_durations__

    def solve_saturate_job_IDs_by_hot_spot(self,
                                           job_ID_to_profit: Dict[str, float],
                                           direct_assignment_plans: Dict[str, 'MMKPScheduler.AssignmentPlan'],
                                           split_assignment_plans: Dict[str, 'MMKPScheduler.AssignmentPlan'],
                                           GPU_comp_mem_capacity: Dict[str, Tuple[int, int]],
                                           sorted_splittable_saturate_job_IDs_: List[str],
                                           in_splittable_saturate_job_IDs_: List[str]
                                           ):
        trail_count_to_solver_result = dict()
        trail_count_to_splitting_jobs: Dict[int, Set[str]] = dict()
        trail_count = 0
        splittable_job_IDs_set = set(sorted_splittable_saturate_job_IDs_)

        optimum_solver_result = self.solve_assignment_plans(assignment_plans_=list(direct_assignment_plans.values()),
                                                            job_ID_to_profit=job_ID_to_profit,
                                                            GPU_comp_mem_capacity=GPU_comp_mem_capacity)
        trail_count_to_solver_result[0] = optimum_solver_result
        trail_count_to_splitting_jobs[0] = set()
        optimum_trail_count = 0

        def get_GPU_remain_resource_diff(solver_res: SolverResult) -> Tuple[Assignments, Dict[str, float]]:
            GPU_type_to_task_comp_mem_requirements: Dict[
                GPUType, Dict[str, Tuple[int, int]]] = {self.GPU_type: dict()}
            for task_ID, data in optimum_solver_result.solver_parameters.task_comp_mem_requirements_and_profits.items():
                GPU_type_to_task_comp_mem_requirements[self.GPU_type][task_ID] = (data[0], data[1])
            assignments_ = Assignments.from_solver_assigment(self.cluster.GPU_ID_to_GPU_type,
                                                             GPU_type_to_task_comp_mem_requirements,
                                                             solver_res.assignment)
            GPU_remain_comp_mem = deepcopy(GPU_comp_mem_capacity)
            for GPU_ID_, task_assignments_ in assignments_.GPU_ID_to_task_assignments.items():
                GPU_type = self.cluster.GPU_ID_to_GPU_type[GPU_ID_]
                GPU_mem = GPUType.normalized_memory(GPU_type=GPU_type)
                for task_assignment_ in task_assignments_:
                    comp, mem = GPU_remain_comp_mem[GPU_ID_]
                    comp -= task_assignment_.comp_req / CompCapacity
                    mem -= task_assignment_.memory / GPU_mem
                    GPU_remain_comp_mem[GPU_ID_] = comp, mem
            GPU_remain_resource_diff_ = dict()
            for GPU_ID_, comp_mem in GPU_remain_comp_mem.items():
                GPU_remain_resource_diff_[GPU_ID_] = abs(comp_mem[0] - comp_mem[1])
            return assignments_, GPU_remain_resource_diff_

        iteration_count = 0
        while True:
            print(f"MMKP scheduler trails for iteration: {iteration_count}")
            assignments, GPU_remain_resource_diff = get_GPU_remain_resource_diff(optimum_solver_result)
            GPU_ID_remain_resource_diff_sorted = sorted(self.cluster.GPU_IDs,
                                                        key=lambda GPU_ID: GPU_remain_resource_diff[GPU_ID],
                                                        reverse=True)
            trail_depth = 0
            trailed_splitting_job_IDs = set()

            stop_trailing = False
            def trail_splitting_job_ID(trail_job_IDs: List[str]):
                nonlocal trail_depth, trail_count, trailed_splitting_job_IDs, stop_trailing
                if set(trail_job_IDs) in trailed_splitting_job_IDs:
                    return
                trail_depth += 1
                trail_count += 1
                trailed_splitting_job_IDs = trailed_splitting_job_IDs.union(set(trail_job_IDs))
                splitting_job_IDs = trail_count_to_splitting_jobs[optimum_trail_count]
                splitting_job_IDs = deepcopy(splitting_job_IDs)
                splitting_job_IDs = splitting_job_IDs.union(trail_job_IDs)
                trail_count_to_splitting_jobs[trail_count] = splitting_job_IDs
                curr_split_assignment_plans = [split_assignment_plans[j_ID] for j_ID in splitting_job_IDs]
                curr_in_split_assignment_plans = [split_assignment_plans[j_ID] for j_ID in splitting_job_IDs if
                                                  j_ID not in splitting_job_IDs] + \
                                                 [direct_assignment_plans[j_ID] for j_ID in
                                                  in_splittable_saturate_job_IDs_]
                assignment_plans = curr_split_assignment_plans + curr_in_split_assignment_plans
                info(
                    f"MMKP scheduler trails with splitting {len(curr_split_assignment_plans)} jobs, in-splitting {len(curr_in_split_assignment_plans)} jobs, ")
                solver_result = self.solve_assignment_plans(
                    assignment_plans_=assignment_plans,
                    job_ID_to_profit=job_ID_to_profit,
                    GPU_comp_mem_capacity=GPU_comp_mem_capacity
                )
                trail_count_to_solver_result[trail_count] = solver_result
                if solver_result.duration / 1e9 > self.solver_duration_upper_bound:
                    stop_trailing = True

            sorted_splittable_job_IDs = sorted_splittable_saturate_job_IDs_[:self.trail_best_splittable_max_depth]
            chunk_size = 2
            groups = [sorted_splittable_job_IDs[i:i + chunk_size] for i in
                      range(0, len(sorted_splittable_job_IDs), chunk_size)]
            for group in groups:
                trail_splitting_job_ID(group)
                if trail_depth > self.exploration_max_depth:
                    break
                if stop_trailing:
                    break
            for GPU_ID in GPU_ID_remain_resource_diff_sorted:
                if stop_trailing:
                    break
                splittable_job_IDs = list()
                for task_assignment in assignments.GPU_ID_to_task_assignments[GPU_ID]:
                    job_ID = task_assignment.task.job_ID
                    if job_ID in splittable_job_IDs_set:
                        splittable_job_IDs.append(job_ID)
                splittable_job_IDs.sort(key=lambda j_ID: split_assignment_plans[j_ID].comprehensive_score)
                groups = [splittable_job_IDs[i:i + chunk_size] for i in range(0, len(splittable_job_IDs), chunk_size)]
                for group in groups:
                    trail_splitting_job_ID(group)
                    if trail_depth > self.exploration_max_depth:
                        break
                    if stop_trailing:
                        break

            max_profit_trail_count = None
            for trail_count, solver_result in trail_count_to_solver_result.items():
                if solver_result.profit > optimum_solver_result.profit:
                    max_profit_trail_count = trail_count
                    print(
                        f"MMKP scheduler trails better, {max_profit_trail_count}, profit: {solver_result.profit}, optimum_solver_result.profit: {optimum_solver_result.profit}")
            if max_profit_trail_count is None:
                break
            else:
                optimum_solver_result = trail_count_to_solver_result[max_profit_trail_count]

        max_profit = None
        max_profit_trail_count = None
        optimum_solver_result__ = None
        solver_durations__ = list()

        for tc, solver_result in trail_count_to_solver_result.items():
            solver_durations__.append(solver_result.duration)
            if max_profit is None or solver_result.profit >= max_profit:
                max_profit = solver_result.profit
                optimum_solver_result__ = solver_result
                max_profit_trail_count = tc
        return optimum_solver_result__, trail_count_to_splitting_jobs[
            max_profit_trail_count], solver_durations__

    @staticmethod
    def build_statistics(solver_durations=None,
                         splitting_job_IDs=None,
                         total_splitting_job_supplied_comp=0,
                         total_splitting_job_supplied_mem=0,
                         job_ID_to_supplied_comp=None,
                         job_ID_to_supplied_mem=None) -> Dict:
        if job_ID_to_supplied_mem is None:
            job_ID_to_supplied_mem = {}
        if job_ID_to_supplied_comp is None:
            job_ID_to_supplied_comp = {}
        if splitting_job_IDs is None:
            splitting_job_IDs = []
        if solver_durations is None:
            solver_durations = []
        return {
            "solver_durations": solver_durations,
            "splitting_job_size": splitting_job_IDs,
            "total_splitting_job_supplied_comp": total_splitting_job_supplied_comp,
            "total_splitting_job_supplied_mem": total_splitting_job_supplied_mem,
            "job_supplied_comp": job_ID_to_supplied_comp,
            "job_supplied_mem": job_ID_to_supplied_mem
        }

    def select_splittable_jobs_by_sorting_balance_score(self,
                                                        preemptive: bool,
                                                        total_normalized_comp: float,
                                                        total_normalized_mem: float,
                                                        total_normalized_consumed_comp: float,
                                                        total_normalized_consumed_mem: float,
                                                        direct_assignment_plans: Dict[
                                                            str, 'MMKPScheduler.AssignmentPlan'],
                                                        split_assignment_plans: Dict[
                                                            str, 'MMKPScheduler.AssignmentPlan'],
                                                        all_job_IDs: List[str]) -> List[str]:
        if len(all_job_IDs) == 0:
            return list()
        saturate_factor = self.non_preemptive_saturate_factor if not preemptive else self.saturate_factor
        total_normalized_remain_comp = total_normalized_comp - total_normalized_consumed_comp
        total_normalized_remain_mem = total_normalized_mem - total_normalized_consumed_mem
        sorted_job_IDs = sorted(all_job_IDs, key=lambda j_ID: direct_assignment_plans[j_ID].balance_score)
        saturate_job_IDs: List[str] = list()
        GPU_mem = GPUType.normalized_memory(self.GPU_type)
        total_consumed_comp = 0
        total_consumed_mem = 0
        for job_ID in sorted_job_IDs:
            assignment_plan = direct_assignment_plans[job_ID]
            total_consumed_comp += assignment_plan.comp_req
            total_consumed_mem += assignment_plan.mem_req
            total_normalized_consumed_resource = total_consumed_mem / GPU_mem + total_consumed_comp / CompCapacity
            if total_normalized_consumed_resource > (
                    total_normalized_remain_comp + total_normalized_remain_mem) * saturate_factor:
                break
            saturate_job_IDs.append(job_ID)

        return saturate_job_IDs

    def extract_splittable_jobs_from_saturate_job_IDs(self,
                                                      split_assignment_plans: Dict[str, 'MMKPScheduler.AssignmentPlan'],
                                                      saturate_job_IDs: List[str]) -> Tuple[List[str], List[str]]:
        splittable_saturate_job_IDs = list()
        in_splittable_saturate_job_IDs = list()
        for saturate_job_ID in saturate_job_IDs:
            if split_assignment_plans[saturate_job_ID] is None or \
                    split_assignment_plans[saturate_job_ID].resource_score > self.resource_score_upper_bound:
                in_splittable_saturate_job_IDs.append(saturate_job_ID)
            else:
                splittable_saturate_job_IDs.append(saturate_job_ID)
        sorted_splittable_saturate_job_IDs = sorted(splittable_saturate_job_IDs,
                                                    key=lambda j_ID: split_assignment_plans[j_ID].comprehensive_score)
        return sorted_splittable_saturate_job_IDs, in_splittable_saturate_job_IDs

    def select_saturate_jobs_by_balancing_total_comp_mem(self,
                                                         preemptive: bool,
                                                         total_normalized_comp: float,
                                                         total_normalized_mem: float,
                                                         total_normalized_consumed_comp: float,
                                                         total_normalized_consumed_mem: float,
                                                         direct_assignment_plans: Dict[
                                                             str, 'MMKPScheduler.AssignmentPlan'],
                                                         split_assignment_plans: Dict[
                                                             str, 'MMKPScheduler.AssignmentPlan'],
                                                         all_job_IDs: List[str]) -> List[str]:
        if len(all_job_IDs) == 0:
            return list()
        saturate_factor = self.non_preemptive_saturate_factor if not preemptive else self.saturate_factor
        saturate_job_IDs: Set[str] = set()
        GPU_mem = GPUType.normalized_memory(self.GPU_type)
        total_normalized_remain_comp = total_normalized_comp - total_normalized_consumed_comp
        total_normalized_remain_mem = total_normalized_mem - total_normalized_consumed_mem
        new_assigned_total_consumed_comp = 0
        new_assigned_total_consumed_mem = 0
        while True:
            diff = np.inf
            selected_job_ID = None
            for job_ID in all_job_IDs:
                if job_ID in saturate_job_IDs:
                    continue
                total_normalized_consumed_comp_ = total_normalized_consumed_comp + direct_assignment_plans[
                    job_ID].comp_req / CompCapacity
                total_normalized_consumed_mem_ = total_normalized_consumed_mem + direct_assignment_plans[
                    job_ID].mem_req / GPU_mem
                diff_ = abs(total_normalized_consumed_comp_ - total_normalized_consumed_mem_)
                if diff_ < diff:
                    diff = diff_
                    selected_job_ID = job_ID
            saturate_job_IDs.add(selected_job_ID)
            total_normalized_consumed_comp += direct_assignment_plans[selected_job_ID].comp_req / CompCapacity
            total_normalized_consumed_mem += direct_assignment_plans[selected_job_ID].mem_req / GPU_mem
            new_assigned_total_consumed_comp += direct_assignment_plans[selected_job_ID].comp_req
            new_assigned_total_consumed_mem += direct_assignment_plans[selected_job_ID].mem_req
            new_assigned_total_normalized_consumed_resource = new_assigned_total_consumed_comp / CompCapacity + new_assigned_total_consumed_mem / GPU_mem
            if new_assigned_total_normalized_consumed_resource > (
                    total_normalized_remain_comp + total_normalized_remain_mem) * saturate_factor:
                break
            if len(saturate_job_IDs) == len(all_job_IDs):
                break

        return list(saturate_job_IDs)

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
            worker_count = assignment_plan.worker_count
            for i in range(assignment_plan.worker_count):
                task = Task(job_ID=job_ID, task_idx=i)
                task_IDs.append(task.task_ID)
                task_comp_mem_requirements_and_profits[task.task_ID] = (
                    assignment_plan.comp_req, assignment_plan.mem_req, job_ID_to_profit[job_ID] / worker_count)
            if worker_count > 1:
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
        _, mem_requirement = self.data_source.get_job_task_memory(
            job_ID=job_ID,
            worker_count=job_spec.plan_worker_count)
        iteration_time = self.data_source.job_iteration_time(
            job_ID=job_ID,
            GPU_type=self.GPU_type,
            comp_req=job_spec.plan_comp,
            worker_count=job_spec.plan_worker_count)
        return MMKPScheduler.AssignmentPlan(
            job_ID=job_ID,
            worker_count=job_spec.plan_worker_count,
            comp_req=job_spec.plan_comp,
            mem_req=mem_requirement,
            GPU_type=self.GPU_type,
            score_weights=self.score_weights,
            direct_plan=None,
            iteration_time=iteration_time)

    def job_splitting_assignment_plan(self, job_ID: str, direct_plan: 'MMKPScheduler.AssignmentPlan') -> Optional[
        'MMKPScheduler.AssignmentPlan']:
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
        iteration_time = None
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
        assert iteration_time is not None
        _, task_mem_requirement = self.data_source.get_job_task_memory(
            job_ID=job_ID,
            worker_count=split_plan_worker_count)
        return MMKPScheduler.AssignmentPlan(job_ID,
                                            split_plan_worker_count,
                                            split_plan_comp,
                                            task_mem_requirement,
                                            self.GPU_type,
                                            self.score_weights,
                                            direct_plan,
                                            iteration_time)

from collections import defaultdict
from typing import Tuple, Optional, List, Dict, Set, Any

import numpy as np

from cluster import TaskAssignment, Assignments
from log import info
from object import GPUType, CompCapacity, Task, Job
from profit import get_profit_calculator
from scheduler import Scheduler
from .RR import RRScheduler
from .solver import SolverParameters, SolverParameters3, do_solve_1, do_solve_3


class MMKPScheduler(Scheduler):
    class ScoreWeights:
        def __init__(self, effective_resource_score_range: Tuple[float, float], balance_weight: float,
                     resource_weight: float):
            self.effective_resource_score_range: Tuple[float, float] = effective_resource_score_range
            self.balance_weight: float = balance_weight
            self.resource_weight: float = resource_weight

    def _init_config(self):
        self.use_split = self.config.get("use_split", True)
        self.strict = self.config.get("strict", True)
        self.throughput_degrade_threshold = self.config.get("throughput_degrade_threshold", 0.03)
        self.GPU_type = GPUType.RTX_2080Ti
        self.plan_comp_unit = self.config.get("plan_comp_unit", 1)
        self.score_weights = [
            MMKPScheduler.ScoreWeights((0, 0.2), 4, 1),
            MMKPScheduler.ScoreWeights((0.2, np.inf), 4, 4)
        ]
        self.resource_score_upper_bound = self.config.get("resource_score_upper_bound", 0.6)
        self.splitting_saturate_factor = self.config.get("splitting_saturate_factor", 0.5)
        self.direct_saturate_factor = self.config.get("direct_saturate_factor", 2.5)
        self.non_preemptive_direct_saturate_factor = self.config.get("non_preemptive_direct_saturate_factor", 128)
        self.non_preemptive_splitting_saturate_factor = self.config.get("non_preemptive_splitting_saturate_factor", 32)
        self.saturate_partitions = self.config.get("saturate_partitions", 5)

        self.direct_saturate_partitions = self.config.get("direct_saturate_partitions", 20)
        self.splitting_saturate_partitions = self.config.get("splitting_saturate_partitions", 5)
        self.trail_best_splittable_max_depth = self.config.get("trail_best_splittable_max_depth", 10)
        self.exploration_max_depth = self.config.get("exploration_max_depth", 20)
        self.use_round_robin_resource_ratio = self.config.get("use_round_robin_resource_ratio", 0.75)
        self.solver_duration_upper_bound = self.config.get("solver_duration_upper_bound", 60)
        self.splitting_jobs_proportion = self.config.get("splitting_jobs_proportion", 1.5)
        self.timeout = self.config.get("timeout", 30)

        self.selector = self.config.get("selector", "balance")

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
            return self.resource_score
            # if self.direct_plan is None:
            #     return 0.
            # resource_score = self.resource_score
            # balance_score = self.balance_score
            # balance_score_increment = balance_score - self.direct_plan.balance_score
            # return balance_score_increment
            # # for score_weight in self.score_weights:
            # #     r = score_weight.effective_resource_score_range
            # #     if r[0] < resource_score < r[1]:
            # #         score = score_weight.balance_weight * balance_score_increment + score_weight.resource_weight * resource_score
            # #         return score
            # assert False

    def use_round_robin(self, all_job_IDs: Set[str], total_normalized_resource: float, preemptive: bool) -> Optional[
        Tuple[Assignments, Optional[Any]]]:
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
            return sche.do_assign(preemptive, 0, set())
        return None

    def reuse_assignments(self):
        return self.cluster.assignments.clone(), MMKPScheduler.build_statistics()

    def do_assign(self, preemptive: bool, now: int, done_jobs_between_preemption: Set[Job]) -> Tuple[
        Assignments, Optional[Any]]:
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
        info(
            f"MMKP starts do assign, preemptive: {preemptive}, done jobs between preemption: {done_jobs_between_preemption}, total_normalized_comp: {total_normalized_comp}, "
            f"total_normalized_mem: {total_normalized_mem}, total_normalized_consumed_comp: {total_normalized_consumed_comp},"
            f"total_normalized_consumed_mem: {total_normalized_consumed_mem}, total_remain_normalized_resource: {total_remain_normalized_resource}")
        use_round_robin_result = self.use_round_robin(all_job_IDs=set(all_job_IDs),
                                                      total_normalized_resource=total_remain_normalized_resource,
                                                      preemptive=preemptive)
        if use_round_robin_result is not None:
            return use_round_robin_result

        profit_calculator = get_profit_calculator(profit_enum=self.profit_enum)

        job_ID_to_profit: Dict[str, float] = profit_calculator.calculate_jobs(data_source=self.data_source,
                                                                              job_IDs=all_job_IDs,
                                                                              GPU_type=self.GPU_type)

        direct_assignment_plans: Dict[str, 'MMKPScheduler.AssignmentPlan'] = dict()
        split_assignment_plans: Dict[str, List['MMKPScheduler.AssignmentPlan']] = dict()
        for job_ID in all_job_IDs:
            direct_plan = self.job_direct_assignment_plan(job_ID=job_ID)
            direct_assignment_plans[job_ID] = direct_plan
            split_assignment_plans_list = self.job_splitting_assignment_plans(job_ID=job_ID, direct_plan=direct_plan)
            if len(split_assignment_plans_list) > 0:
                split_assignment_plans[job_ID] = split_assignment_plans_list

        if self.selector == "random":
            selector = self.select_saturate_jobs_by_random
        elif self.selector == "balance":
            selector = self.select_saturate_jobs_by_balancing_total_comp_mem
        else:
            assert False
        saturate_direct_job_IDs = selector(
            preemptive=preemptive,
            total_normalized_consumed_comp=total_normalized_consumed_comp,
            total_normalized_consumed_mem=total_normalized_consumed_mem,
            total_normalized_comp=total_normalized_comp,
            total_normalized_mem=total_normalized_mem,
            direct_assignment_plans=direct_assignment_plans,
            split_assignment_plans=split_assignment_plans,
            all_job_IDs=all_job_IDs, splitting=False)
        sorted_splittable_assignment_plans, in_splittable_job_IDs = self.extract_split_well_plans_from_job_IDs(
            split_assignment_plans, saturate_direct_job_IDs)
        if len(saturate_direct_job_IDs) > 10:
            splitting_saturate_ratio = int(self.direct_saturate_factor / self.splitting_saturate_factor)
            sorted_splittable_assignment_plans = sorted_splittable_assignment_plans[:
                                                                                    min(len(
                                                                                        sorted_splittable_assignment_plans) // splitting_saturate_ratio,
                                                                                        len(sorted_splittable_assignment_plans))]
        else:
            sorted_splittable_assignment_plans = sorted_splittable_assignment_plans
        optimum_solver_result, splitting_plans, task_comp_mem_requirements, solver_durations, timeout_count = self.solve_saturate_job_IDs_by_MMKP_2(
            preemptive,
            job_ID_to_profit,
            direct_assignment_plans,
            sorted_splittable_assignment_plans,
            GPU_comp_mem_capacity,
            saturate_direct_job_IDs,
            in_splittable_job_IDs)

        if optimum_solver_result is None:
            return self.cluster.assignments.clone(), MMKPScheduler.build_statistics()
        optimum_assignment = optimum_solver_result.assignment
        GPU_type_to_task_comp_mem_requirements: Dict[
            GPUType, Dict[str, Tuple[int, int]]] = {self.GPU_type: task_comp_mem_requirements}
        assignments = Assignments.from_solver_assigment(GPU_ID_to_GPU_type=self.cluster.GPU_ID_to_GPU_type,
                                                        GPU_type_to_task_comp_mem_requirements=GPU_type_to_task_comp_mem_requirements,
                                                        solver_assignments=optimum_assignment)
        if not preemptive:
            assignments = self.cluster.assignments.merge(assignments)
        if not self.strict:
            GPU_ID_to_task_assignments = assignments.GPU_ID_to_task_assignments
            assigned_jobs = set(assignments.job_ID_to_task_assignments.keys())
            unassigned_jobs = [job_ID for job_ID in all_job_IDs if job_ID not in assigned_jobs]
            assignments = RRScheduler.assign_jobs(self.strict,
                                                  self.data_source,
                                                  unassigned_jobs,
                                                  self.cluster.GPU_IDs,
                                                  self.GPU_type,
                                                  GPU_ID_to_task_assignments)

        total_splitting_job_supplied_comp = 0
        total_splitting_job_supplied_mem = 0
        job_ID_to_supplied_comp = dict()
        job_ID_to_supplied_mem = dict()
        for splitting_assignment_plan in splitting_plans:
            assert isinstance(splitting_assignment_plan, MMKPScheduler.AssignmentPlan)
            splitting_job_ID = splitting_assignment_plan.job_ID
            if splitting_job_ID not in assignments.job_ID_to_task_assignments:
                continue
            direct_assignment_plan = direct_assignment_plans[splitting_job_ID]
            supplied_comp = splitting_assignment_plan.worker_count * splitting_assignment_plan.comp_req - direct_assignment_plan.comp_req * direct_assignment_plan.worker_count
            total_splitting_job_supplied_comp += supplied_comp
            job_ID_to_supplied_comp[splitting_job_ID] = supplied_comp
            supplied_mem = splitting_assignment_plan.mem_req * splitting_assignment_plan.worker_count - direct_assignment_plan.mem_req * direct_assignment_plan.worker_count
            total_splitting_job_supplied_mem += supplied_mem
            job_ID_to_supplied_mem[splitting_job_ID] = supplied_mem
        # assignments = assignments.supplement_over_supply()
        statistics = MMKPScheduler.build_statistics(timeout_count=timeout_count,
                                                    solver_durations=solver_durations,
                                                    splitting_plans=splitting_plans,
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
            timeout=self.timeout,
            assignment_plans=assignment_plans_,
            job_ID_to_profit=partial_job_ID_to_profit_,
            GPU_comp_mem_capacity=GPU_comp_mem_capacity)
        return do_solve_1(solver_params=solver_parameters_)

    def solve_assignment_plans_3(self,
                                 direct_assignment_plans: List['MMKPScheduler.AssignmentPlan'],
                                 splitting_assignment_plans: List['MMKPScheduler.AssignmentPlan'],
                                 job_ID_to_profit: Dict[str, float],
                                 GPU_comp_mem_capacity: Dict[str, Tuple[int, int]]):
        #     solver_type: SolverEnum
        #     GPU_type: GPUType
        #     dist_job_to_tasks: Dict[str, Tuple[str, ...]]
        #     GPU_comp_mem_capacity: Dict[str, Tuple[int, int]]
        #     task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, Union[int, float]]]
        splitting_job_IDs = set()
        for assignment_plan in splitting_assignment_plans:
            splitting_job_IDs.add(assignment_plan.job_ID)

        job_ID_to_fixed_task_IDs_lists = defaultdict(list)
        fixed_task_ID_to_job_ID = dict()
        splitting_job_ID_task_sets: Dict[str, List[List[str]]] = defaultdict(list)
        for assignment_plan in splitting_assignment_plans:
            job_ID = assignment_plan.job_ID
            fixed_task_IDs = list()
            for i in range(assignment_plan.worker_count):
                task = Task(job_ID=job_ID, task_idx=i)
                fixed_task_ID = f"splitting_true|{task.task_ID}"
                fixed_task_IDs.append(fixed_task_ID)
                fixed_task_ID_to_job_ID[fixed_task_ID] = job_ID
            job_ID_to_fixed_task_IDs_lists[assignment_plan.job_ID].append((fixed_task_IDs, assignment_plan))
            splitting_job_ID_task_sets[assignment_plan.job_ID].append(fixed_task_IDs)
        for assignment_plan in direct_assignment_plans:
            job_ID = assignment_plan.job_ID
            fixed_task_IDs = list()
            for i in range(assignment_plan.worker_count):
                task = Task(job_ID=job_ID, task_idx=i)
                if job_ID in splitting_job_IDs:
                    fixed_task_ID = f"splitting_false|{task.task_ID}"
                else:
                    fixed_task_ID = task.task_ID
                fixed_task_IDs.append(fixed_task_ID)
                fixed_task_ID_to_job_ID[fixed_task_ID] = job_ID
            job_ID_to_fixed_task_IDs_lists[assignment_plan.job_ID].append((fixed_task_IDs, assignment_plan))
            if assignment_plan.job_ID in splitting_job_IDs:
                splitting_job_ID_task_sets[assignment_plan.job_ID].append(fixed_task_IDs)

        task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, float]] = dict()
        dist_tasks: List[Tuple[str, ...]] = list()
        for job_ID, fixed_task_IDs_assignment_plans_list in job_ID_to_fixed_task_IDs_lists.items():
            fixed_task_IDs_of_curr_job = list()
            for item in fixed_task_IDs_assignment_plans_list:
                fixed_task_IDs, assignment_plan = item
                fixed_task_IDs_of_curr_job.extend(fixed_task_IDs)
                for task_ID in fixed_task_IDs:
                    task_comp_mem_requirements_and_profits[task_ID] = (
                        assignment_plan.comp_req, assignment_plan.mem_req,
                        job_ID_to_profit[job_ID] / len(fixed_task_IDs))
                if len(fixed_task_IDs) > 1:
                    dist_tasks.append(tuple(fixed_task_IDs))

        solver_parameters = SolverParameters3(
            timeout=self.timeout,
            splitting_job_ID_task_sets=splitting_job_ID_task_sets,
            solver_type=self.solver_enum,
            GPU_type=self.GPU_type,
            dist_tasks=dist_tasks,
            GPU_comp_mem_capacity=GPU_comp_mem_capacity,
            task_comp_mem_requirements_and_profits=task_comp_mem_requirements_and_profits
        )
        solver_result = do_solve_3(solver_params=solver_parameters)
        if solver_result is None:
            info("MMKP do solve 3 result is None, timeout.")
            return None, None

        def fixed_task_ID_to_original(fixed_task_ID_: str):
            if fixed_task_ID_.startswith("splitting"):
                return fixed_task_ID_.split("|", 1)[-1]
            return fixed_task_ID_

        task_comp_mem = dict()
        reverted_solver_assignment = defaultdict(set)
        for GPU_ID, fixed_task_IDs in solver_result.assignment.items():
            for fixed_task_ID in fixed_task_IDs:
                comp_mem_profit = task_comp_mem_requirements_and_profits[fixed_task_ID]
                task_ID = fixed_task_ID_to_original(fixed_task_ID)
                task_comp_mem[task_ID] = comp_mem_profit[0], comp_mem_profit[1]
                reverted_solver_assignment[GPU_ID].add(fixed_task_ID_to_original(fixed_task_ID))
        solver_result.assignment = reverted_solver_assignment
        return solver_result, task_comp_mem

    def solve_saturate_job_IDs_by_partitions(self,
                                             preemptive: bool,
                                             job_ID_to_profit: Dict[str, float],
                                             direct_assignment_plans: Dict[str, 'MMKPScheduler.AssignmentPlan'],
                                             split_assignment_plans: Dict[str, 'MMKPScheduler.AssignmentPlan'],
                                             GPU_comp_mem_capacity: Dict[str, Tuple[int, int]],
                                             sorted_splittable_saturate_job_IDs_: List[str],
                                             in_splittable_saturate_job_IDs_: List[str]):
        trail_count_to_solver_result = dict()
        trail_count_to_splitting_jobs = dict()
        total_splittable_job_size = len(sorted_splittable_saturate_job_IDs_)
        if preemptive:
            partition_size = 1
        else:
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
            if len(splitting_job_IDs__) > self.splitting_jobs_proportion * len(direct_job_IDs):
                info(
                    f"MMKP scheduler, skip rest partitions due to splitting too much, accumulated_partition_size: {accumulated_partition_size}")
                break
            assignment_plans = \
                [split_assignment_plans[j_ID] for j_ID in splitting_job_IDs__] + \
                [direct_assignment_plans[j_ID] for j_ID in direct_job_IDs]
            trail_count_to_splitting_jobs[trail_count] = splitting_job_IDs__
            solver_result = self.solve_assignment_plans(
                assignment_plans_=assignment_plans,
                job_ID_to_profit=job_ID_to_profit,
                GPU_comp_mem_capacity=GPU_comp_mem_capacity
            )
            trail_count_to_solver_result[trail_count] = solver_result
            trail_count += 1
            accumulated_partition_size += partition_size
            if solver_result.duration / 1e9 > self.solver_duration_upper_bound:
                info(
                    f"MMKP scheduler, skip rest partitions due to too much time consuming, accumulated_partition_size: {accumulated_partition_size}")
                break
        max_profit = None
        max_profit_trail_count = None
        optimum_solver_result__ = None
        solver_durations__ = list()

        for tc, solver_result in trail_count_to_solver_result.items():
            solver_durations__.append(solver_result.duration)
            if max_profit is None or solver_result.profit > max_profit:
                max_profit = solver_result.profit
                optimum_solver_result__ = solver_result
                max_profit_trail_count = tc
        task_comp_mem = dict()
        for task_ID, comp_mem_profits in optimum_solver_result__.solver_parameters.task_comp_mem_requirements_and_profits.items():
            task_comp_mem[task_ID] = comp_mem_profits[0], comp_mem_profits[1]
        return optimum_solver_result__, trail_count_to_splitting_jobs[
            max_profit_trail_count], task_comp_mem, solver_durations__

    def solve_saturate_job_IDs_by_MMKP_2(self,
                                         preemptive: bool,
                                         job_ID_to_profit: Dict[str, float],
                                         direct_assignment_plans: Dict[str, 'MMKPScheduler.AssignmentPlan'],
                                         sorted_splittable_assignment_plans: List['MMKPScheduler.AssignmentPlan'],
                                         GPU_comp_mem_capacity: Dict[str, Tuple[int, int]],
                                         saturate_direct_job_IDs: List[str],
                                         in_splittable_saturate_job_IDs_: List[str]):
        solver_durations = list()

        MMKP_1_solver_result = None
        MMKP_1_splitting_jobs = None
        MMKP_1_task_comp_mem = None
        timeout_count = 0
        for i in range(1000):
            # try direct solve, which is fast
            direct_partition_size = len(saturate_direct_job_IDs) // self.direct_saturate_partitions
            if direct_partition_size == 0:
                direct_partition_size = 1
            saturate_direct_job_IDs_ = saturate_direct_job_IDs[
                                       :max(0, len(saturate_direct_job_IDs) - i * direct_partition_size)]
            MMKP_1_assignment_plans = [direct_assignment_plans[job_ID] for job_ID in saturate_direct_job_IDs_]
            info(
                f"MMKP scheduler start solving all direct assignment plans, plans size: {len(MMKP_1_assignment_plans)}, try count: {i}")
            MMKP_1_solver_result = self.solve_assignment_plans(
                assignment_plans_=MMKP_1_assignment_plans,
                job_ID_to_profit=job_ID_to_profit,
                GPU_comp_mem_capacity=GPU_comp_mem_capacity)
            if MMKP_1_solver_result is None:
                info(
                    f"MMKP scheduler solves direct assignment failed due to timeout, try count: {i}")
                timeout_count += 1
                continue
            info(
                f"MMKP scheduler solves all direct assignment plans with: {MMKP_1_solver_result.duration / 1e9}, try count {i}")
            MMKP_1_splitting_jobs = list()
            solver_durations.append(MMKP_1_solver_result.duration)
            MMKP_1_task_comp_mem = dict()
            for task_ID, comp_mem_profits in MMKP_1_solver_result.solver_parameters.task_comp_mem_requirements_and_profits.items():
                MMKP_1_task_comp_mem[task_ID] = comp_mem_profits[0], comp_mem_profits[1]
            break
        if MMKP_1_solver_result.profit == float(len(self.cluster.GPU_IDs) * 2):
            info(
                f"MMKP scheduler direct find optimal profit, use direct only.")
            return MMKP_1_solver_result, MMKP_1_splitting_jobs, MMKP_1_task_comp_mem, solver_durations, timeout_count

        if not self.use_split:
            return MMKP_1_solver_result, MMKP_1_splitting_jobs, MMKP_1_task_comp_mem, solver_durations, timeout_count

        MMKP_2_solver_result = None
        MMKP_2_splitting_plans = None
        MMKP_2_task_comp_mem = None
        for i in range(1000):
            info(f"MMKP scheduler is trying MMKP 2, try count: {i}")
            # try MMKP 2

            split_partition_size = len(sorted_splittable_assignment_plans) // self.splitting_saturate_partitions
            if split_partition_size == 0:
                split_partition_size = 1
            sorted_splittable_assignment_plans_ = sorted_splittable_assignment_plans[:max(
                len(sorted_splittable_assignment_plans) - i * split_partition_size, 0)]

            if len(sorted_splittable_assignment_plans_) == 0:
                # 先减split的
                direct_partition_size = len(saturate_direct_job_IDs) // self.direct_saturate_partitions
                if direct_partition_size == 0:
                    direct_partition_size = 1
                saturate_direct_job_IDs_ = saturate_direct_job_IDs[
                                           :max(0, len(saturate_direct_job_IDs) - i * direct_partition_size)]
            else:
                saturate_direct_job_IDs_ = saturate_direct_job_IDs[:]

            MMKP_2_direct_assignment_plans = [direct_assignment_plans[job_ID] for job_ID in saturate_direct_job_IDs_]

            MMKP_2_splitting_assignment_plans = sorted_splittable_assignment_plans_

            info(
                f"MMKP scheduler solving all direct and splitting assignment plans with MMKP_2, direct plans size: {len(MMKP_2_direct_assignment_plans)}, splitting plans size: {len(MMKP_2_splitting_assignment_plans)}, try count: {i}")
            MMKP_2_solver_result, MMKP_2_task_comp_mem = self.solve_assignment_plans_3(
                direct_assignment_plans=MMKP_2_direct_assignment_plans,
                splitting_assignment_plans=MMKP_2_splitting_assignment_plans,
                job_ID_to_profit=job_ID_to_profit,
                GPU_comp_mem_capacity=GPU_comp_mem_capacity
            )
            if MMKP_2_solver_result is None or MMKP_2_task_comp_mem is None:
                info(
                    f"MMKP scheduler solves all direct and splitting assignment failed due to timeout, try count: {i}")
                timeout_count += 1
                continue
            info(
                f"MMKP scheduler solves all direct assignment plans with: {MMKP_2_solver_result.duration / 1e9}, try count: {i}")
            task_max_idx = defaultdict(int)
            MMKP_2_splitting_plans = list()
            for task_ID in MMKP_2_task_comp_mem:
                task = Task.from_task_ID(task_ID)
                task_max_idx[task] = max(task_max_idx[task], task.task_idx)
            splitting_work_counts = defaultdict(set)
            job_ID_to_worker_count_to_assignment_plan = defaultdict(dict)
            for plan in MMKP_2_splitting_assignment_plans:
                splitting_work_counts[plan.job_ID].add(plan.worker_count)
                job_ID_to_worker_count_to_assignment_plan[plan.job_ID][plan.worker_count] = plan
            for task, max_idx in task_max_idx.items():
                worker_count = (max_idx + 1)
                if task.job_ID in splitting_work_counts and \
                        worker_count in splitting_work_counts[task.job_ID]:
                    assignment_plan = job_ID_to_worker_count_to_assignment_plan[task.job_ID][worker_count]
                    MMKP_2_splitting_plans.append(assignment_plan)

            MMKP_2_splitting_plans = list(MMKP_2_splitting_plans)
            solver_durations.append(MMKP_2_solver_result.duration)
            break

        MMKP_1_wins = MMKP_1_solver_result.profit >= MMKP_2_solver_result.profit
        optimum_solver_result = MMKP_1_solver_result if MMKP_1_wins else MMKP_2_solver_result
        splitting_job_IDs = MMKP_1_splitting_jobs if MMKP_1_wins else MMKP_2_splitting_plans
        task_comp_mem = MMKP_1_task_comp_mem if MMKP_1_wins else MMKP_2_task_comp_mem
        return optimum_solver_result, splitting_job_IDs, task_comp_mem, solver_durations, timeout_count

    # def solve_saturate_job_IDs_by_hot_spot(self,
    #                                        job_ID_to_profit: Dict[str, float],
    #                                        direct_assignment_plans: Dict[str, 'MMKPScheduler.AssignmentPlan'],
    #                                        split_assignment_plans: Dict[str, 'MMKPScheduler.AssignmentPlan'],
    #                                        GPU_comp_mem_capacity: Dict[str, Tuple[int, int]],
    #                                        sorted_splittable_saturate_job_IDs_: List[str],
    #                                        in_splittable_saturate_job_IDs_: List[str]
    #                                        ):
    #     trail_count_to_solver_result = dict()
    #     trail_count_to_splitting_jobs: Dict[int, Set[str]] = dict()
    #     trail_count = 0
    #     splittable_job_IDs_set = set(sorted_splittable_saturate_job_IDs_)
    #
    #     optimum_solver_result = self.solve_assignment_plans(assignment_plans_=list(direct_assignment_plans.values()),
    #                                                         job_ID_to_profit=job_ID_to_profit,
    #                                                         GPU_comp_mem_capacity=GPU_comp_mem_capacity)
    #     trail_count_to_solver_result[0] = optimum_solver_result
    #     trail_count_to_splitting_jobs[0] = set()
    #     optimum_trail_count = 0
    #
    #     def get_GPU_remain_resource_diff(solver_res: SolverResult) -> Tuple[Assignments, Dict[str, float]]:
    #         GPU_type_to_task_comp_mem_requirements: Dict[
    #             GPUType, Dict[str, Tuple[int, int]]] = {self.GPU_type: dict()}
    #         for task_ID, data in optimum_solver_result.solver_parameters.task_comp_mem_requirements_and_profits.items():
    #             GPU_type_to_task_comp_mem_requirements[self.GPU_type][task_ID] = (data[0], data[1])
    #         assignments_ = Assignments.from_solver_assigment(self.cluster.GPU_ID_to_GPU_type,
    #                                                          GPU_type_to_task_comp_mem_requirements,
    #                                                          solver_res.assignment)
    #         GPU_remain_comp_mem = deepcopy(GPU_comp_mem_capacity)
    #         for GPU_ID_, task_assignments_ in assignments_.GPU_ID_to_task_assignments.items():
    #             GPU_type = self.cluster.GPU_ID_to_GPU_type[GPU_ID_]
    #             GPU_mem = GPUType.normalized_memory(GPU_type=GPU_type)
    #             for task_assignment_ in task_assignments_:
    #                 comp, mem = GPU_remain_comp_mem[GPU_ID_]
    #                 comp -= task_assignment_.comp_req / CompCapacity
    #                 mem -= task_assignment_.memory / GPU_mem
    #                 GPU_remain_comp_mem[GPU_ID_] = comp, mem
    #         GPU_remain_resource_diff_ = dict()
    #         for GPU_ID_, comp_mem in GPU_remain_comp_mem.items():
    #             GPU_remain_resource_diff_[GPU_ID_] = abs(comp_mem[0] - comp_mem[1])
    #         return assignments_, GPU_remain_resource_diff_
    #
    #     iteration_count = 0
    #     while True:
    #         print(f"MMKP scheduler trails for iteration: {iteration_count}")
    #         assignments, GPU_remain_resource_diff = get_GPU_remain_resource_diff(optimum_solver_result)
    #         GPU_ID_remain_resource_diff_sorted = sorted(self.cluster.GPU_IDs,
    #                                                     key=lambda GPU_ID: GPU_remain_resource_diff[GPU_ID],
    #                                                     reverse=True)
    #         trail_depth = 0
    #         trailed_splitting_job_IDs = set()
    #
    #         stop_trailing = False
    #
    #         def trail_splitting_job_ID(trail_job_IDs: List[str]):
    #             nonlocal trail_depth, trail_count, trailed_splitting_job_IDs, stop_trailing
    #             if set(trail_job_IDs) in trailed_splitting_job_IDs:
    #                 return
    #             trail_depth += 1
    #             trail_count += 1
    #             trailed_splitting_job_IDs = trailed_splitting_job_IDs.union(set(trail_job_IDs))
    #             splitting_job_IDs = trail_count_to_splitting_jobs[optimum_trail_count]
    #             splitting_job_IDs = deepcopy(splitting_job_IDs)
    #             splitting_job_IDs = splitting_job_IDs.union(trail_job_IDs)
    #             trail_count_to_splitting_jobs[trail_count] = splitting_job_IDs
    #             curr_split_assignment_plans = [split_assignment_plans[j_ID] for j_ID in splitting_job_IDs]
    #             curr_in_split_assignment_plans = [split_assignment_plans[j_ID] for j_ID in splitting_job_IDs if
    #                                               j_ID not in splitting_job_IDs] + \
    #                                              [direct_assignment_plans[j_ID] for j_ID in
    #                                               in_splittable_saturate_job_IDs_]
    #             assignment_plans = curr_split_assignment_plans + curr_in_split_assignment_plans
    #             info(
    #                 f"MMKP scheduler trails with splitting {len(curr_split_assignment_plans)} jobs, in-splitting {len(curr_in_split_assignment_plans)} jobs, ")
    #             solver_result = self.solve_assignment_plans(
    #                 assignment_plans_=assignment_plans,
    #                 job_ID_to_profit=job_ID_to_profit,
    #                 GPU_comp_mem_capacity=GPU_comp_mem_capacity
    #             )
    #             trail_count_to_solver_result[trail_count] = solver_result
    #             if solver_result.duration / 1e9 > self.solver_duration_upper_bound:
    #                 stop_trailing = True
    #
    #         sorted_splittable_job_IDs = sorted_splittable_saturate_job_IDs_[:self.trail_best_splittable_max_depth]
    #         chunk_size = 2
    #         groups = [sorted_splittable_job_IDs[i:i + chunk_size] for i in
    #                   range(0, len(sorted_splittable_job_IDs), chunk_size)]
    #         for group in groups:
    #             trail_splitting_job_ID(group)
    #             if trail_depth > self.exploration_max_depth:
    #                 break
    #             if stop_trailing:
    #                 break
    #         for GPU_ID in GPU_ID_remain_resource_diff_sorted:
    #             if stop_trailing:
    #                 break
    #             splittable_job_IDs = list()
    #             for task_assignment in assignments.GPU_ID_to_task_assignments[GPU_ID]:
    #                 job_ID = task_assignment.task.job_ID
    #                 if job_ID in splittable_job_IDs_set:
    #                     splittable_job_IDs.append(job_ID)
    #             splittable_job_IDs.sort(key=lambda j_ID: split_assignment_plans[j_ID].comprehensive_score)
    #             groups = [splittable_job_IDs[i:i + chunk_size] for i in range(0, len(splittable_job_IDs), chunk_size)]
    #             for group in groups:
    #                 trail_splitting_job_ID(group)
    #                 if trail_depth > self.exploration_max_depth:
    #                     break
    #                 if stop_trailing:
    #                     break
    #
    #         max_profit_trail_count = None
    #         for trail_count, solver_result in trail_count_to_solver_result.items():
    #             if solver_result.profit > optimum_solver_result.profit:
    #                 max_profit_trail_count = trail_count
    #                 print(
    #                     f"MMKP scheduler trails better, {max_profit_trail_count}, profit: {solver_result.profit}, optimum_solver_result.profit: {optimum_solver_result.profit}")
    #         if max_profit_trail_count is None:
    #             break
    #         else:
    #             optimum_solver_result = trail_count_to_solver_result[max_profit_trail_count]
    #
    #     max_profit = None
    #     max_profit_trail_count = None
    #     optimum_solver_result__ = None
    #     solver_durations__ = list()
    #
    #     for tc, solver_result in trail_count_to_solver_result.items():
    #         solver_durations__.append(solver_result.duration)
    #         if max_profit is None or solver_result.profit >= max_profit:
    #             max_profit = solver_result.profit
    #             optimum_solver_result__ = solver_result
    #             max_profit_trail_count = tc
    #     return optimum_solver_result__, trail_count_to_splitting_jobs[
    #         max_profit_trail_count], solver_durations__

    @staticmethod
    def build_statistics(timeout_count=None,
                         solver_durations=None,
                         splitting_plans=None,
                         total_splitting_job_supplied_comp=0,
                         total_splitting_job_supplied_mem=0,
                         job_ID_to_supplied_comp=None,
                         job_ID_to_supplied_mem=None) -> Dict:
        if job_ID_to_supplied_mem is None:
            job_ID_to_supplied_mem = {}
        if job_ID_to_supplied_comp is None:
            job_ID_to_supplied_comp = {}
        if splitting_plans is None:
            splitting_plans = []
        if solver_durations is None:
            solver_durations = []
        splitting_job_worker_counts = {splitting_plan.job_ID: splitting_plan.worker_count for splitting_plan in
                                       splitting_plans}
        return {
            "timeout_count": timeout_count,
            "solver_durations": solver_durations,
            "splitting_job_worker_counts": splitting_job_worker_counts,
            "total_splitting_job_supplied_comp": total_splitting_job_supplied_comp,
            "total_splitting_job_supplied_mem": total_splitting_job_supplied_mem,
            "job_supplied_comp": job_ID_to_supplied_comp,
            "job_supplied_mem": job_ID_to_supplied_mem
        }

    # def select_splittable_jobs_by_sorting_balance_score(self,
    #                                                     preemptive: bool,
    #                                                     total_normalized_comp: float,
    #                                                     total_normalized_mem: float,
    #                                                     total_normalized_consumed_comp: float,
    #                                                     total_normalized_consumed_mem: float,
    #                                                     direct_assignment_plans: Dict[
    #                                                         str, 'MMKPScheduler.AssignmentPlan'],
    #                                                     split_assignment_plans: Dict[
    #                                                         str, 'MMKPScheduler.AssignmentPlan'],
    #                                                     all_job_IDs: List[str]) -> List[str]:
    #     if len(all_job_IDs) == 0:
    #         return list()
    #     saturate_factor = self.non_preemptive_saturate_factor if not preemptive else self.saturate_factor
    #     total_normalized_remain_comp = total_normalized_comp - total_normalized_consumed_comp
    #     total_normalized_remain_mem = total_normalized_mem - total_normalized_consumed_mem
    #     sorted_job_IDs = sorted(all_job_IDs, key=lambda j_ID: direct_assignment_plans[j_ID].balance_score)
    #     saturate_job_IDs: List[str] = list()
    #     GPU_mem = GPUType.normalized_memory(self.GPU_type)
    #     total_consumed_comp = 0
    #     total_consumed_mem = 0
    #     for job_ID in sorted_job_IDs:
    #         assignment_plan = direct_assignment_plans[job_ID]
    #         total_consumed_comp += assignment_plan.comp_req
    #         total_consumed_mem += assignment_plan.mem_req
    #         total_normalized_consumed_resource = total_consumed_mem / GPU_mem + total_consumed_comp / CompCapacity
    #         if total_normalized_consumed_resource > (
    #                 total_normalized_remain_comp + total_normalized_remain_mem) * saturate_factor:
    #             break
    #         saturate_job_IDs.append(job_ID)
    #
    #     return saturate_job_IDs

    def extract_split_well_plans_from_job_IDs(self,
                                              split_assignment_plans: Dict[str, List['MMKPScheduler.AssignmentPlan']],
                                              job_IDs: List[str]) -> Tuple[
        List['MMKPScheduler.AssignmentPlan'], List[str]]:
        in_splittable_saturate_job_IDs = list()
        splittable_plans: List['MMKPScheduler.AssignmentPlan'] = list()
        for job_ID in job_IDs:
            if job_ID not in split_assignment_plans:
                in_splittable_saturate_job_IDs.append(job_ID)
                continue
            in_splittable = True
            for splittable_assignment_plan in split_assignment_plans[job_ID]:
                if splittable_assignment_plan.resource_score < self.resource_score_upper_bound:
                    splittable_plans.append(splittable_assignment_plan)
                    in_splittable = False
            if in_splittable:
                continue
        splittable_plans = sorted(splittable_plans,
                                  key=lambda plan: plan.comprehensive_score)
        return splittable_plans, in_splittable_saturate_job_IDs

    def select_saturate_jobs_by_balancing_total_comp_mem(self,
                                                         preemptive: bool,
                                                         total_normalized_comp: float,
                                                         total_normalized_mem: float,
                                                         total_normalized_consumed_comp: float,
                                                         total_normalized_consumed_mem: float,
                                                         direct_assignment_plans: Dict[
                                                             str, 'MMKPScheduler.AssignmentPlan'],
                                                         split_assignment_plans: Dict[
                                                             str, List['MMKPScheduler.AssignmentPlan']],
                                                         all_job_IDs: List[str],
                                                         splitting: bool
                                                         ) -> List[str]:
        if splitting:
            all_job_IDs = list(split_assignment_plans.keys())
        if len(all_job_IDs) == 0:
            return list()
        if splitting:
            saturate_factor = self.splitting_saturate_factor
        else:
            saturate_factor = self.direct_saturate_factor
        if not preemptive:
            if splitting:
                saturate_factor = self.non_preemptive_splitting_saturate_factor
            else:
                saturate_factor = self.non_preemptive_direct_saturate_factor
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

    def select_saturate_jobs_by_random(self,
                                       preemptive: bool,
                                       total_normalized_comp: float,
                                       total_normalized_mem: float,
                                       total_normalized_consumed_comp: float,
                                       total_normalized_consumed_mem: float,
                                       direct_assignment_plans: Dict[
                                           str, 'MMKPScheduler.AssignmentPlan'],
                                       split_assignment_plans: Dict[
                                           str, List['MMKPScheduler.AssignmentPlan']],
                                       all_job_IDs: List[str],
                                       splitting: bool
                                       ) -> List[str]:
        if splitting:
            all_job_IDs = list(split_assignment_plans.keys())
        if len(all_job_IDs) == 0:
            return list()
        if splitting:
            saturate_factor = self.splitting_saturate_factor
        else:
            saturate_factor = self.direct_saturate_factor
        if not preemptive:
            if splitting:
                saturate_factor = self.non_preemptive_splitting_saturate_factor
            else:
                saturate_factor = self.non_preemptive_direct_saturate_factor
        saturate_job_IDs: Set[str] = set()
        GPU_mem = GPUType.normalized_memory(self.GPU_type)
        total_normalized_remain_comp = total_normalized_comp - total_normalized_consumed_comp
        total_normalized_remain_mem = total_normalized_mem - total_normalized_consumed_mem
        new_assigned_total_consumed_comp = 0
        new_assigned_total_consumed_mem = 0
        all_job_IDs = list(all_job_IDs)
        np.random.shuffle(all_job_IDs)
        for job_ID in all_job_IDs:
            if job_ID in saturate_job_IDs:
                continue
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
                                timeout: int,
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
            timeout=timeout,
            strict=self.strict,
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

    def job_splitting_assignment_plans(self, job_ID: str, direct_plan: 'MMKPScheduler.AssignmentPlan') -> List[
        'MMKPScheduler.AssignmentPlan']:
        splitting_assignment_plan_2 = self.job_splitting_assignment_plan_by_factor(job_ID=job_ID,
                                                                                   direct_plan=direct_plan,
                                                                                   split_factor=2)
        splitting_assignment_plan_4 = self.job_splitting_assignment_plan_by_factor(job_ID=job_ID,
                                                                                   direct_plan=direct_plan,
                                                                                   split_factor=4)
        plans = list(filter(lambda item: item is not None, [splitting_assignment_plan_2, splitting_assignment_plan_4]))
        return plans

    def job_splitting_assignment_plan_by_factor(self, job_ID: str, direct_plan: 'MMKPScheduler.AssignmentPlan',
                                                split_factor: int) -> Optional[
        'MMKPScheduler.AssignmentPlan']:
        job_spec = self.data_source.get_job_spec(job_ID=job_ID)
        original_iteration_time = self.data_source.job_iteration_time(job_ID=job_ID, GPU_type=self.GPU_type,
                                                                      comp_req=job_spec.plan_comp,
                                                                      worker_count=job_spec.plan_worker_count)
        split_plan_worker_count = split_factor * job_spec.plan_worker_count
        if job_spec.plan_comp % split_factor == 0:
            split_plan_comp = job_spec.plan_comp // split_factor
        else:
            split_plan_comp = job_spec.plan_comp // split_factor + 1
        if split_plan_comp == 0:
            split_plan_comp = 1
        no_suitable_splitting = False
        iteration_time = None
        while True:
            iteration_time = self.data_source.job_iteration_time(
                job_ID=job_ID,
                GPU_type=self.GPU_type,
                comp_req=split_plan_comp,
                worker_count=split_plan_worker_count)
            if iteration_time is None:
                return None
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

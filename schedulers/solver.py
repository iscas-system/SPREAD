import math
import time
from collections import defaultdict
from itertools import count
from typing import Dict, Tuple, Union, Optional, Set, List

import gurobipy as gu
from gurobipy import GRB

from log import *
from model import SolverResult, SolverParameters, SolverParameters2, SolverParameters3, SolverParameters4, \
    PartitionSolverParameters, \
    PartitionSolverResult, JobDistributionSolverParameters, JobDistributionSolverResult
from object import SolverEnum


def get_solver(solver_enum: 'SolverEnum'):
    return {
        SolverEnum.MMKP: AssignmentSolver.MMKP_1
    }[solver_enum]


def do_MMKP_solve_1(solver_params: SolverParameters) -> Optional[SolverResult]:
    # info(f"received solver parameters, solver_params: {solver_params}")
    solver = get_solver(solver_params.solver_type)
    start = time.time_ns()
    solve_raw_res = AssignmentSolver.MMKP_1(
        timeout=solver_params.timeout,
        strict=solver_params.strict, dist_job_to_tasks=solver_params.dist_job_to_tasks,
        GPU_comp_mem_capacity=solver_params.GPU_comp_mem_capacity,
        task_comp_mem_requirements_and_profits=solver_params.task_comp_mem_requirements_and_profits)
    end = time.time_ns()
    if solve_raw_res is None:
        return None
    assignment, profit = solve_raw_res
    solver_result = SolverResult(solver_parameters=solver_params, duration=(end - start), profit=profit,
                                 assignment=assignment)
    return solver_result


def do_MMKP_solve_2(solver_params: SolverParameters2) -> Optional[SolverResult]:
    # info(f"received solver parameters, solver_params: {solver_params}")
    start = time.time_ns()
    solve_raw_res = AssignmentSolver.MMKP_2(
        timeout=solver_params.timeout,
        splitting_task_IDs_list_list=solver_params.splitting_task_IDs_list_list,
        dist_job_to_tasks=solver_params.dist_job_to_tasks,
        GPU_comp_mem_capacity=solver_params.GPU_comp_mem_capacity,
        task_comp_mem_requirements_and_profits=solver_params.task_comp_mem_requirements_and_profits)
    end = time.time_ns()
    if solve_raw_res is None:
        return None
    assignment, profit = solve_raw_res
    solver_result = SolverResult(solver_parameters2=solver_params, duration=(end - start), profit=profit,
                                 assignment=assignment)
    return solver_result


def do_MMKP_solve_3(solver_params: SolverParameters3) -> Optional[SolverResult]:
    # info(f"received solver parameters, solver_params: {solver_params}")
    start = time.time_ns()
    solve_raw_res = AssignmentSolver.MMKP_3(
        timeout=solver_params.timeout,
        splitting_job_ID_task_sets=solver_params.splitting_job_ID_task_sets,
        dist_tasks=solver_params.dist_tasks,
        GPU_comp_mem_capacity=solver_params.GPU_comp_mem_capacity,
        task_comp_mem_requirements_and_profits=solver_params.task_comp_mem_requirements_and_profits)
    end = time.time_ns()
    if solve_raw_res is None:
        return None
    assignment, profit = solve_raw_res
    solver_result = SolverResult(solver_parameters3=solver_params, duration=(end - start), profit=profit,
                                 assignment=assignment)
    return solver_result


def do_MMKP_solve_4(solver_params: SolverParameters4) -> Optional[SolverResult]:
    # info(f"received solver parameters, solver_params: {solver_params}")
    start = time.time_ns()
    solve_raw_res = AssignmentSolver.MMKP_4(
        timeout=solver_params.timeout,
        job_ID_to_spread_job_IDs=solver_params.job_ID_to_spread_job_IDs,
        spread_job_ID_to_task_sets=solver_params.spread_job_ID_to_task_sets,
        GPU_comp_mem_capacity=solver_params.GPU_comp_mem_capacity,
        in_node_spread_job_IDs=solver_params.in_node_spread_job_IDs,
        cross_node_spread_job_IDs=solver_params.cross_node_spread_job_IDs,
        task_comp_mem_requirements_and_profits=solver_params.task_comp_mem_requirements_and_profits,
        GPU_ID_to_node_id=solver_params.GPU_ID_to_node_id)
    end = time.time_ns()
    if solve_raw_res is None:
        return None
    assignment, profit = solve_raw_res
    solver_result = SolverResult(solver_parameters4=solver_params, duration=(end - start), profit=profit,
                                 assignment=assignment)
    return solver_result


def do_partition_solve_1(solver_params: PartitionSolverParameters) -> Optional[PartitionSolverResult]:
    start = time.time_ns()
    solve_raw_res = PartitionSolver.solve(
        GPU_ID_to_node_id=solver_params.GPU_ID_to_node_id,
        partition_size=solver_params.partition_size,
        strategy=solver_params.strategy
    )
    end = time.time_ns()
    if solve_raw_res is None:
        return None
    partition_to_GPU_IDs, GPU_ID_to_partition, partition_profit = solve_raw_res
    solver_result = PartitionSolverResult(solver_parameters=solver_params,
                                          duration=(end - start),
                                          GPU_ID_to_partition=GPU_ID_to_partition,
                                          partition_to_GPU_IDs=partition_to_GPU_IDs,
                                          partition_profit=partition_profit)
    return solver_result


def do_job_distribution_solve_1(solver_params: JobDistributionSolverParameters) -> Optional[
    JobDistributionSolverResult]:
    start = time.time_ns()
    solve_raw_res = JobDistributionSolver.solve_heuristic(
        partition_to_GPU_IDs=solver_params.partition_to_GPU_IDs,
        GPU_comp_mem_total_capacity=solver_params.GPU_comp_mem_total_capacity,
        GPU_comp_mem_capacity=solver_params.GPU_comp_mem_capacity,
        job_comp_mem_demand=solver_params.job_comp_mem_demand,
        job_priority=solver_params.job_priority,
        strategy=solver_params.strategy
    )
    end = time.time_ns()
    if solve_raw_res is None:
        return None
    partition_to_jobs = solve_raw_res
    solver_result = JobDistributionSolverResult(solver_parameters=solver_params,
                                                duration=(end - start),
                                                partition_to_jobs=partition_to_jobs)
    return solver_result


class PartitionSolver:
    @staticmethod
    def solve(
            GPU_ID_to_node_id: Dict[str, str],
            partition_size: int,
            strategy: str = "heuristic"  # "heuristic", "round"
    ):
        GPU_IDs_ = sorted(list(GPU_ID_to_node_id.keys()))
        if len(GPU_IDs_) % partition_size != 0:
            virtual_counter = count(0)
            while len(GPU_IDs_) % partition_size != 0:
                GPU_ID_ = f"virtual_{next(virtual_counter)}"
                GPU_IDs_.append(GPU_ID_)
                GPU_ID_to_node_id[GPU_ID_] = "virtual_node_id"
        node_id_to_GPU_IDs = defaultdict(set)
        for GPU_ID_, node_id_ in GPU_ID_to_node_id.items():
            node_id_to_GPU_IDs[node_id_].add(GPU_ID_)

        partitions = [f"partition_{i}" for i in range(0, math.ceil(len(GPU_IDs_) / partition_size))]

        partition_to_GPU_IDs = defaultdict(list)

        def heuristic_strategy():
            for p in partitions:
                while len(partition_to_GPU_IDs[p]) != partition_size:
                    remaining_GPU_count = partition_size - len(partition_to_GPU_IDs[p])
                    target_node_id = None
                    closest_GPU_diff = None
                    for node_id, GPU_IDs in node_id_to_GPU_IDs.items():
                        if len(GPU_IDs) == 0:
                            continue
                        GPU_count_diff = abs(len(GPU_IDs) - remaining_GPU_count)
                        if closest_GPU_diff is None or GPU_count_diff < closest_GPU_diff:
                            closest_GPU_diff = GPU_count_diff
                            target_node_id = node_id
                        if GPU_count_diff == closest_GPU_diff and len(GPU_IDs) < len(
                                node_id_to_GPU_IDs[target_node_id]):
                            target_node_id = node_id

                    remaining_GPU_IDs = sorted(list(node_id_to_GPU_IDs[target_node_id]))
                    taken_GPU_count = min(remaining_GPU_count, len(remaining_GPU_IDs))
                    taken_GPU_IDs = remaining_GPU_IDs[:taken_GPU_count]
                    for GPU_ID in taken_GPU_IDs:
                        node_id_to_GPU_IDs[target_node_id].discard(GPU_ID)
                    partition_to_GPU_IDs[p].extend(taken_GPU_IDs)

        def round_strategy():
            for p in partitions:
                while len(partition_to_GPU_IDs[p]) != partition_size:
                    remaining_GPU_count = partition_size - len(partition_to_GPU_IDs[p])
                    target_node_id = None
                    for node_id, GPU_IDs in node_id_to_GPU_IDs.items():
                        if len(GPU_IDs) == 0:
                            continue
                        target_node_id = node_id
                        break

                    remaining_GPU_IDs = sorted(list(node_id_to_GPU_IDs[target_node_id]))
                    taken_GPU_count = min(remaining_GPU_count, len(remaining_GPU_IDs))
                    taken_GPU_IDs = remaining_GPU_IDs[:taken_GPU_count]
                    for GPU_ID in taken_GPU_IDs:
                        node_id_to_GPU_IDs[target_node_id].discard(GPU_ID)
                    partition_to_GPU_IDs[p].extend(taken_GPU_IDs)

        strategy_to_func = {
            "heuristic": heuristic_strategy,
            "round": round_strategy
        }
        strategy_to_func[strategy]()
        for p_, GPU_IDs_ in partition_to_GPU_IDs.items():
            partition_to_GPU_IDs[p_] = [a for a in GPU_IDs_ if "virtual" not in a]

        GPU_ID_to_partition = dict()
        for p_, GPU_IDs_ in partition_to_GPU_IDs.items():
            for a in GPU_IDs_:
                GPU_ID_to_partition[a] = p_
        partition_profit = cal_partition_profit(partition_to_GPU_IDs, GPU_ID_to_node_id)
        return partition_to_GPU_IDs, GPU_ID_to_partition, partition_profit


class AssignmentSolver:
    @staticmethod
    def MMKP_1(
            timeout: int,
            strict: bool,
            dist_job_to_tasks: Dict[str, Tuple[str, ...]],
            GPU_comp_mem_capacity: Dict[str, Tuple[int, int]],
            task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, Union[int, float]]]) \
            -> Optional[Tuple[Dict[str, Set[str]], Union[int, float]]]:
        if len(task_comp_mem_requirements_and_profits) == 0:
            return dict(), 0
        dist_job_task_size_tasks = {
            dist_job: (len(tasks), tasks) for dist_job, tasks in dist_job_to_tasks.items()
        }
        GPUs, GPU_comp_capacity, GPU_mem_capacity = gu.multidict(GPU_comp_mem_capacity)
        # GPUs, GPU_comp_capacity, GPU_mem_capacity = gu.multidict({
        #     "T4_1": (10, 15),
        #     "T4_2": (10, 15)
        # })
        tasks, task_comp_requirements, task_mem_requirements, task_profits = gu.multidict(
            task_comp_mem_requirements_and_profits)
        # tasks, task_comp_requirements, task_mem_requirements, task_profits = gu.multidict({
        #     "task_1_job_1": [5, 5, 8],
        #     "task_2_job_1": [5, 5, 8],
        #     "task_1_job_2": [3, 10, 13],
        #     "task_1_job_3": [2, 5, 7],
        # })

        m = gu.Model()
        X = m.addVars(tasks, GPUs, vtype=GRB.BINARY)

        m.addConstrs(
            (X.sum(t, '*') <= 1 for t in tasks),
            "each_task_appears_only_once")

        m.addConstrs(
            (gu.quicksum(task_comp_requirements[t] * X[t, a] for t in tasks) <= GPU_comp_capacity[a]
             for a in GPUs),
            "task_comp_requirement_less_than_GPU_capacity")

        m.addConstrs(
            (gu.quicksum(task_mem_requirements[t] * X[t, a] for t in tasks) <= GPU_mem_capacity[a]
             for a in GPUs),
            "task_mem_requirement_less_than_GPU_capacity")

        if len(dist_job_task_size_tasks) > 0:
            dist_jobs, dist_job_task_size, dist_job_tasks = gu.multidict(dist_job_task_size_tasks)
            # dist_jobs, dist_job_task_size, dist_job_tasks = gu.multidict({
            #     "job_1": (2, ("task_1_job_1", "task_2_job_1"))
            # })
            m.addConstrs(
                (gu.quicksum(X[t, a]
                             for t in dist_job_tasks[dist_job]) <= 1
                 for a in GPUs
                 for dist_job in dist_jobs),
                "dist_job_tasks_not_on_same_GPU")

            z = gu.tupledict()
            for dist_job in dist_jobs:
                z[dist_job, 1] = m.addVar(vtype=GRB.BINARY, name=f"{dist_job}_indicator_1")
                z[dist_job, 2] = m.addVar(vtype=GRB.BINARY, name=f"{dist_job}_indicator_2")

                m.addConstr(z[dist_job, 1] + z[dist_job, 2] == 1, f"{dist_job}_indicator_equation")

                m.addGenConstrIndicator(z[dist_job, 1], True,
                                        gu.quicksum(X[t, a] for t in dist_job_tasks[dist_job] for a in GPUs),
                                        GRB.LESS_EQUAL, 0,
                                        name=f"{dist_job}_indicator_LESS_EQUAL_0")
                m.addGenConstrIndicator(z[dist_job, 2], True,
                                        gu.quicksum(X[t, a] for t in dist_job_tasks[dist_job] for a in GPUs),
                                        GRB.GREATER_EQUAL,
                                        dist_job_task_size[dist_job],
                                        name=f"{dist_job}_indicator_GREATER_EQUAL_0")

        m.setObjective(gu.quicksum(task_profits[t] * X[t, a] for t in tasks for a in GPUs), GRB.MAXIMIZE)
        m.setParam('TimeLimit', timeout)
        m.update()
        start = time.time_ns()
        m.optimize()
        end = time.time_ns()

        if m.Status != GRB.OPTIMAL:
            info(f"MMKP solver finds unexpected none optimal solution, status = {m.Status}")
            return None

        info(
            f"MMKP solver finds optimal solution, objective value == {m.ObjVal}, duration seconds = {(end - start) / 1e9}")
        assignment: Dict[str, Set[str]] = defaultdict(set)
        for a in GPUs:
            for t in tasks:
                if X[t, a].X < 0.5:
                    continue
                assignment[a].add(t)
        own_calculated_profit = float(cal_profit(task_comp_mem_requirements_and_profits, assignment))
        diff = abs(m.ObjVal - own_calculated_profit)
        info(f"diff with own calculated profit: {diff}")
        if diff > 1e-7:
            info(f"diff > 1e-7: {diff}")

        return assignment, cal_profit(task_comp_mem_requirements_and_profits, assignment)

    @staticmethod
    def MMKP_2(
            timeout: int,
            splitting_task_IDs_list_list: List[List[str]],
            dist_job_to_tasks: Dict[str, Tuple[str, ...]],
            GPU_comp_mem_capacity: Dict[str, Tuple[int, int]],
            task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, Union[int, float]]]) \
            -> Optional[Tuple[Dict[str, Set[str]], Union[int, float]]]:
        # original_job_ID_to_total_task_size = dict()
        # splitting_job_ID_to_original_job_ID = dict()
        # for original_job_ID, splitting_jobs_sets in original_job_ID_to_splitting_job_IDs.items():
        #     for splitting_jobs in splitting_jobs_sets:
        #         for splitting_job in splitting_jobs:
        #             splitting_job_ID_to_original_job_ID[splitting_job] = original_job_ID
        #             if splitting_job in dist_job_to_tasks:
        #                 original_job_ID_to_total_task_size[original_job_ID] += len(dist_job_to_tasks[splitting_job])
        #             else:
        #                 original_job_ID_to_total_task_size[original_job_ID] += 1
        if len(task_comp_mem_requirements_and_profits) == 0:
            return dict(), 0

        dist_job_task_size_tasks = {
            dist_job: (len(tasks), tasks) for dist_job, tasks in dist_job_to_tasks.items()
        }
        GPUs, GPU_comp_capacity, GPU_mem_capacity = gu.multidict(GPU_comp_mem_capacity)
        # GPUs, GPU_comp_capacity, GPU_mem_capacity = gu.multidict({
        #     "T4_1": (10, 15),
        #     "T4_2": (10, 15)
        # })
        tasks, task_comp_requirements, task_mem_requirements, task_profits = gu.multidict(
            task_comp_mem_requirements_and_profits)
        # tasks, task_comp_requirements, task_mem_requirements, task_profits = gu.multidict({
        #     "task_1_job_1": [5, 5, 8],
        #     "task_2_job_1": [5, 5, 8],
        #     "task_1_job_2": [3, 10, 13],
        #     "task_1_job_3": [2, 5, 7],
        # })

        m = gu.Model()
        X = m.addVars(tasks, GPUs, vtype=GRB.BINARY)

        m.addConstrs(
            (X.sum(t, '*') <= 1 for t in tasks),
            "each_task_appears_only_once")

        m.addConstrs(
            (gu.quicksum(task_comp_requirements[t] * X[t, a] for t in tasks) <= GPU_comp_capacity[a]
             for a in GPUs),
            "task_comp_requirement_less_than_GPU_capacity")

        m.addConstrs(
            (gu.quicksum(task_mem_requirements[t] * X[t, a] for t in tasks) <= GPU_mem_capacity[a]
             for a in GPUs),
            "task_mem_requirement_less_than_GPU_capacity")

        if len(dist_job_task_size_tasks) > 0:
            dist_jobs, dist_job_task_size, dist_job_tasks = gu.multidict(dist_job_task_size_tasks)
            # dist_jobs, dist_job_task_size, dist_job_tasks = gu.multidict({
            #     "job_1": (2, ("task_1_job_1", "task_2_job_1"))
            # })
            m.addConstrs(
                (gu.quicksum(X[t, a]
                             for t in dist_job_tasks[dist_job]) <= 1
                 for a in GPUs
                 for dist_job in dist_jobs),
                "dist_job_tasks_not_on_same_GPU")

            z = gu.tupledict()
            constr_added_splitting_original_job_IDs = set()
            for dist_job in dist_jobs:
                z[dist_job, 1] = m.addVar(vtype=GRB.BINARY, name=f"{dist_job}_indicator_1")
                z[dist_job, 2] = m.addVar(vtype=GRB.BINARY, name=f"{dist_job}_indicator_2")

                m.addConstr(z[dist_job, 1] + z[dist_job, 2] == 1, f"{dist_job}_indicator_equation")

                m.addGenConstrIndicator(z[dist_job, 1], True,
                                        gu.quicksum(X[t, a] for t in dist_job_tasks[dist_job] for a in GPUs),
                                        GRB.LESS_EQUAL, 0,
                                        name=f"{dist_job}_indicator_LESS_EQUAL_0")
                m.addGenConstrIndicator(z[dist_job, 2], True,
                                        gu.quicksum(X[t, a] for t in dist_job_tasks[dist_job] for a in GPUs),
                                        GRB.GREATER_EQUAL,
                                        dist_job_task_size[dist_job],
                                        name=f"{dist_job}_indicator_GREATER_EQUAL_0")

        if len(splitting_task_IDs_list_list) > 0:
            for splitting_task_IDs_list in splitting_task_IDs_list_list:
                size = len(splitting_task_IDs_list)
                m.addConstr(gu.quicksum(X[t, a] for t in splitting_task_IDs_list for a in GPUs) <= size - 1,
                            "splitting_tasks_not_co_exist")

        m.setObjective(gu.quicksum(task_profits[t] * X[t, a] for t in tasks for a in GPUs), GRB.MAXIMIZE)
        m.setParam('TimeLimit', timeout)
        m.update()
        start = time.time_ns()
        m.optimize()
        end = time.time_ns()

        if m.Status != GRB.OPTIMAL:
            info(f"MMKP solver finds unexpected none optimal solution, status = {m.Status}")
            return None

        info(
            f"MMKP solver finds optimal solution, objective value == {m.ObjVal}, duration seconds = {(end - start) / 1e9}")
        assignment: Dict[str, Set[str]] = defaultdict(set)
        for a in GPUs:
            for t in tasks:
                if X[t, a].X < 0.5:
                    continue
                assignment[a].add(t)
        own_calculated_profit = float(cal_profit(task_comp_mem_requirements_and_profits, assignment))
        diff = abs(m.ObjVal - own_calculated_profit)
        info(f"diff with own calculated profit: {diff}")
        if diff > 1e-7:
            info(f"diff > 1e-7: {diff}")

        assigned_tasks = set()
        for tasks in assignment.values():
            assigned_tasks.update(tasks)
        for splitting_task_IDs_list in splitting_task_IDs_list_list:
            diff = set(splitting_task_IDs_list).difference(assigned_tasks)
            assert len(diff) > 0
        return assignment, cal_profit(task_comp_mem_requirements_and_profits, assignment)

    @staticmethod
    def MMKP_3(
            timeout: int,
            splitting_job_ID_task_sets: Dict[str, List[List[str]]],
            dist_tasks: List[Tuple[str, ...]],
            GPU_comp_mem_capacity: Dict[str, Tuple[int, int]],
            task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, Union[int, float]]]) \
            -> Optional[Tuple[Dict[str, Set[str]], Union[int, float]]]:
        # original_job_ID_to_total_task_size = dict()
        # splitting_job_ID_to_original_job_ID = dict()
        # for original_job_ID, splitting_jobs_sets in original_job_ID_to_splitting_job_IDs.items():
        #     for splitting_jobs in splitting_jobs_sets:
        #         for splitting_job in splitting_jobs:
        #             splitting_job_ID_to_original_job_ID[splitting_job] = original_job_ID
        #             if splitting_job in dist_job_to_tasks:
        #                 original_job_ID_to_total_task_size[original_job_ID] += len(dist_job_to_tasks[splitting_job])
        #             else:
        #                 original_job_ID_to_total_task_size[original_job_ID] += 1
        if len(task_comp_mem_requirements_and_profits) == 0:
            return dict(), 0

        dist_job_task_size_tasks = {
            f"dist_job_{idx}": (len(tasks), tasks) for idx, tasks in enumerate(dist_tasks)
        }
        GPUs, GPU_comp_capacity, GPU_mem_capacity = gu.multidict(GPU_comp_mem_capacity)
        # GPUs, GPU_comp_capacity, GPU_mem_capacity = gu.multidict({
        #     "T4_1": (10, 15),
        #     "T4_2": (10, 15)
        # })
        tasks, task_comp_requirements, task_mem_requirements, task_profits = gu.multidict(
            task_comp_mem_requirements_and_profits)
        # tasks, task_comp_requirements, task_mem_requirements, task_profits = gu.multidict({
        #     "task_1_job_1": [5, 5, 8],
        #     "task_2_job_1": [5, 5, 8],
        #     "task_1_job_2": [3, 10, 13],
        #     "task_1_job_3": [2, 5, 7],
        # })

        m = gu.Model()
        X = m.addVars(tasks, GPUs, vtype=GRB.BINARY)

        m.addConstrs(
            (X.sum(t, '*') <= 1 for t in tasks),
            "each_task_appears_only_once")

        m.addConstrs(
            (gu.quicksum(task_comp_requirements[t] * X[t, a] for t in tasks) <= GPU_comp_capacity[a]
             for a in GPUs),
            "task_comp_requirement_less_than_GPU_capacity")

        m.addConstrs(
            (gu.quicksum(task_mem_requirements[t] * X[t, a] for t in tasks) <= GPU_mem_capacity[a]
             for a in GPUs),
            "task_mem_requirement_less_than_GPU_capacity")

        if len(dist_job_task_size_tasks) > 0:
            dist_jobs, dist_job_task_size, dist_job_tasks = gu.multidict(dist_job_task_size_tasks)
            # dist_jobs, dist_job_task_size, dist_job_tasks = gu.multidict({
            #     "job_1": (2, ("task_1_job_1", "task_2_job_1"))
            # })
            m.addConstrs(
                (gu.quicksum(X[t, a]
                             for t in dist_job_tasks[dist_job]) <= 1
                 for a in GPUs
                 for dist_job in dist_jobs),
                "dist_job_tasks_not_on_same_GPU")

            z = gu.tupledict()
            for dist_job in dist_jobs:
                z[dist_job, 1] = m.addVar(vtype=GRB.BINARY, name=f"{dist_job}_indicator_1")
                z[dist_job, 2] = m.addVar(vtype=GRB.BINARY, name=f"{dist_job}_indicator_2")

                m.addConstr(z[dist_job, 1] + z[dist_job, 2] == 1, f"{dist_job}_indicator_equation")

                m.addGenConstrIndicator(z[dist_job, 1], True,
                                        gu.quicksum(X[t, a] for t in dist_job_tasks[dist_job] for a in GPUs),
                                        GRB.LESS_EQUAL, 0,
                                        name=f"{dist_job}_indicator_LESS_EQUAL_0")
                m.addGenConstrIndicator(z[dist_job, 2], True,
                                        gu.quicksum(X[t, a] for t in dist_job_tasks[dist_job] for a in GPUs),
                                        GRB.GREATER_EQUAL,
                                        dist_job_task_size[dist_job],
                                        name=f"{dist_job}_indicator_GREATER_EQUAL_0")

        if len(splitting_job_ID_task_sets) > 0:

            sp = gu.tupledict()

            for splitting_job, split_plans_lists in splitting_job_ID_task_sets.items():
                for i, splitting_plan_task_IDs in enumerate(split_plans_lists):
                    sp[splitting_job, i] = m.addVar(vtype=GRB.BINARY, name=f"{splitting_job}_indicator_{i}")

                    # m.addGenConstrIndicator(sp[splitting_job, i], True,
                    #                     gu.quicksum(X[t, a] for t in splitting_plan_task_IDs for a in GPUs) >= 1,
                    #                     name=f"{splitting_job}_indicator_GREATER_EQUAL_1")
                    m.addConstr(
                        (sp[splitting_job, i] == 1)
                        >>
                        (gu.quicksum(X[t, a] for t in splitting_plan_task_IDs for a in GPUs) == 0)
                    )
                m.addConstr(gu.quicksum(sp.select(splitting_job, '*')) >= len(split_plans_lists) - 1,
                            f"{splitting_job}_indicator_equation")

        m.setObjective(gu.quicksum(task_profits[t] * X[t, a] for t in tasks for a in GPUs), GRB.MAXIMIZE)
        m.setParam('TimeLimit', timeout)
        m.update()
        start = time.time_ns()
        m.optimize()
        end = time.time_ns()

        if m.Status != GRB.OPTIMAL:
            info(f"MMKP solver finds unexpected none optimal solution, status = {m.Status}")
            return None

        info(
            f"MMKP solver finds optimal solution, objective value == {m.ObjVal}, duration seconds = {(end - start) / 1e9}")
        assignment: Dict[str, Set[str]] = defaultdict(set)
        for a in GPUs:
            for t in tasks:
                if X[t, a].X < 0.5:
                    continue
                assignment[a].add(t)
        own_calculated_profit = float(cal_profit(task_comp_mem_requirements_and_profits, assignment))
        diff = abs(m.ObjVal - own_calculated_profit)
        info(f"diff with own calculated profit: {diff}")
        if diff > 1e-7:
            info(f"diff > 1e-7: {diff}")

        assigned_tasks = set()
        for tasks in assignment.values():
            assigned_tasks.update(tasks)
        return assignment, cal_profit(task_comp_mem_requirements_and_profits, assignment)

    @staticmethod
    def MMKP_4(
            timeout: int,
            job_ID_to_spread_job_IDs: Dict[str, List[str]],
            spread_job_ID_to_task_sets: Dict[str, List[str]],
            GPU_comp_mem_capacity: Dict[str, Tuple[int, int]],
            task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, Union[int, float]]],
            in_node_spread_job_IDs: List[str],
            cross_node_spread_job_IDs: List[str],
            GPU_ID_to_node_id: Dict[str, str]) \
            -> Optional[Tuple[Dict[str, Set[str]], Union[int, float]]]:

        if len(task_comp_mem_requirements_and_profits) == 0:
            return dict(), 0
        node_id_to_GPU_IDs = defaultdict(list)
        for GPU_ID, node_id in GPU_ID_to_node_id.items():
            node_id_to_GPU_IDs[node_id].append(GPU_ID)

        if len(node_id_to_GPU_IDs) == 1:
            for job_ID, spread_job_IDs in job_ID_to_spread_job_IDs.items():
                for cross_node_spread_job_ID in cross_node_spread_job_IDs:
                    if cross_node_spread_job_ID in spread_job_IDs:
                        spread_job_IDs.remove(cross_node_spread_job_ID)
            for cross_node_spread_job_ID in cross_node_spread_job_IDs:
                task_sets = spread_job_ID_to_task_sets[cross_node_spread_job_ID]
                for task in task_sets:
                    task_comp_mem_requirements_and_profits.pop(task)
                spread_job_ID_to_task_sets.pop(cross_node_spread_job_ID)

        dist_tasks = list()
        for spread_job_ID, task_sets in spread_job_ID_to_task_sets.items():
            if len(task_sets) > 1:
                dist_tasks.append(task_sets)

        dist_job_task_size_tasks = {
            f"dist_job_{idx}": (len(tasks), tasks) for idx, tasks in enumerate(dist_tasks)
        }
        GPUs, GPU_comp_capacity, GPU_mem_capacity = gu.multidict(GPU_comp_mem_capacity)
        # GPUs, GPU_comp_capacity, GPU_mem_capacity = gu.multidict({
        #     "T4_1": (10, 15),
        #     "T4_2": (10, 15)
        # })
        tasks, task_comp_requirements, task_mem_requirements, task_profits = gu.multidict(
            task_comp_mem_requirements_and_profits)
        # tasks, task_comp_requirements, task_mem_requirements, task_profits = gu.multidict({
        #     "task_1_job_1": [5, 5, 8],
        #     "task_2_job_1": [5, 5, 8],
        #     "task_1_job_2": [3, 10, 13],
        #     "task_1_job_3": [2, 5, 7],
        # })

        m = gu.Model()
        X = m.addVars(tasks, GPUs, vtype=GRB.BINARY)

        m.addConstrs(
            (X.sum(t, '*') <= 1 for t in tasks),
            "each_task_appears_only_once")

        m.addConstrs(
            (gu.quicksum(task_comp_requirements[t] * X[t, a] for t in tasks) <= GPU_comp_capacity[a]
             for a in GPUs),
            "task_comp_requirement_less_than_GPU_capacity")

        m.addConstrs(
            (gu.quicksum(task_mem_requirements[t] * X[t, a] for t in tasks) <= GPU_mem_capacity[a]
             for a in GPUs),
            "task_mem_requirement_less_than_GPU_capacity")

        if len(dist_job_task_size_tasks) > 0:
            dist_jobs, dist_job_task_size, dist_job_tasks = gu.multidict(dist_job_task_size_tasks)
            # dist_jobs, dist_job_task_size, dist_job_tasks = gu.multidict({
            #     "job_1": (2, ("task_1_job_1", "task_2_job_1"))
            # })
            m.addConstrs(
                (gu.quicksum(X[t, a]
                             for t in dist_job_tasks[dist_job]) <= 1
                 for a in GPUs
                 for dist_job in dist_jobs),
                "dist_job_tasks_not_on_same_GPU")

            z = gu.tupledict()
            for dist_job in dist_jobs:
                z[dist_job, 1] = m.addVar(vtype=GRB.BINARY, name=f"{dist_job}_indicator_1")
                z[dist_job, 2] = m.addVar(vtype=GRB.BINARY, name=f"{dist_job}_indicator_2")

                m.addConstr(z[dist_job, 1] + z[dist_job, 2] == 1, f"{dist_job}_indicator_equation")

                m.addGenConstrIndicator(z[dist_job, 1], True,
                                        gu.quicksum(X[t, a] for t in dist_job_tasks[dist_job] for a in GPUs),
                                        GRB.LESS_EQUAL, 0,
                                        name=f"{dist_job}_indicator_LESS_EQUAL_0")
                m.addGenConstrIndicator(z[dist_job, 2], True,
                                        gu.quicksum(X[t, a] for t in dist_job_tasks[dist_job] for a in GPUs),
                                        GRB.GREATER_EQUAL,
                                        dist_job_task_size[dist_job],
                                        name=f"{dist_job}_indicator_GREATER_EQUAL_0")

        sp = gu.tupledict()

        for job_ID, spread_job_IDs in job_ID_to_spread_job_IDs.items():
            if len(spread_job_IDs) == 1:
                continue
            for i, spread_job_ID in enumerate(spread_job_IDs):
                task_sets = spread_job_ID_to_task_sets[spread_job_ID]
                sp[job_ID, i] = m.addVar(vtype=GRB.BINARY, name=f"{job_ID}_spread_only_one_indicator_{i}")

                # m.addGenConstrIndicator(sp[splitting_job, i], True,
                #                     gu.quicksum(X[t, a] for t in splitting_plan_task_IDs for a in GPUs) >= 1,
                #                     name=f"{splitting_job}_indicator_GREATER_EQUAL_1")
                m.addConstr(
                    (sp[job_ID, i] == 1)
                    >>
                    (gu.quicksum(X[t, a] for t in task_sets for a in GPUs) == 0)
                )
            m.addConstr(gu.quicksum(sp.select(job_ID, '*')) >= len(spread_job_IDs) - 1,
                        f"{job_ID}_spread_only_one_indicator_equation")

        if len(node_id_to_GPU_IDs) > 1:
            # in node
            in_node = gu.tupledict()
            for in_node_spread_job_ID in in_node_spread_job_IDs:
                spread_tasks = spread_job_ID_to_task_sets[in_node_spread_job_ID]
                if len(spread_tasks) == 1:
                    continue
                for node_id in node_id_to_GPU_IDs.keys():
                    GPU_IDs_of_node = node_id_to_GPU_IDs[node_id]
                    in_node[in_node_spread_job_ID, 1] = m.addVar(vtype=GRB.BINARY,
                                                                 name=f"in_node_{in_node_spread_job_ID}_indicator_1")
                    in_node[in_node_spread_job_ID, 2] = m.addVar(vtype=GRB.BINARY,
                                                                 name=f"in_node_{in_node_spread_job_ID}_indicator_2")

                    m.addConstr(in_node[in_node_spread_job_ID, 1] + in_node[in_node_spread_job_ID, 2] == 1,
                                f"in_node_{in_node_spread_job_ID}_indicator_equation")

                    m.addGenConstrIndicator(in_node[in_node_spread_job_ID, 1], True,
                                            gu.quicksum(X[t, a] for t in spread_tasks for a in
                                                        GPU_IDs_of_node),
                                            GRB.LESS_EQUAL, 0,
                                            name=f"in_node_{in_node_spread_job_ID}_indicator_LESS_EQUAL_0")
                    m.addGenConstrIndicator(in_node[in_node_spread_job_ID, 2], True,
                                            gu.quicksum(X[t, a] for t in spread_tasks for a in
                                                        GPU_IDs_of_node),
                                            GRB.GREATER_EQUAL,
                                            len(spread_tasks),
                                            name=f"in_node_{in_node_spread_job_ID}_indicator_GREATER_EQUAL_0")

            # cross node
            cross_node = gu.tupledict()
            cross_node_sum = gu.tupledict()
            for cross_node_spread_job_ID in cross_node_spread_job_IDs:
                spread_tasks = spread_job_ID_to_task_sets[cross_node_spread_job_ID]
                if len(spread_tasks) == 1:
                    continue
                for node_id in node_id_to_GPU_IDs.keys():
                    GPU_IDs_of_node = node_id_to_GPU_IDs[node_id]
                    m.addConstr(gu.quicksum(X[t, a] for t in spread_tasks for a in GPU_IDs_of_node) <= len(spread_tasks) - 1, f"cross_node_{cross_node_spread_job_ID}_job_not_on_same_node")
                #
                #     cross_node[cross_node_spread_job_ID, node_id] = m.addVar(vtype=GRB.BINARY,
                #                                                              name=f"{cross_node_spread_job_ID}_on_{node_id}_only_one_indicator")
                #     m.addConstr(
                #         (cross_node[cross_node_spread_job_ID, node_id] == 0)
                #         >>
                #         (gu.quicksum(X[t, a] for t in spread_tasks for a in GPU_IDs_of_node) <= len(spread_tasks) - 1)
                #     )
                #     m.addConstr(
                #         (cross_node[cross_node_spread_job_ID, node_id] == 1)
                #         >>
                #         (gu.quicksum(X[t, a] for t in spread_tasks for a in GPU_IDs_of_node) == len(spread_tasks))
                #     )
                # m.addConstr(
                #         (gu.quicksum(cross_node.select(cross_node_spread_job_ID, '*')) == 0),
                #         f"cross_node_{cross_node_spread_job_ID}_indicator_equation")

                # cross_node_sum[cross_node_spread_job_ID, 1] = m.addVar(vtype=GRB.BINARY,
                #                                                        name=f"cross_node_{cross_node_spread_job_ID}_indicator_1")
                # cross_node_sum[cross_node_spread_job_ID, 2] = m.addVar(vtype=GRB.BINARY,
                #                                                        name=f"cross_node_{cross_node_spread_job_ID}_indicator_2")

                # m.addConstr(
                #     cross_node_sum[cross_node_spread_job_ID, 1] + cross_node_sum[cross_node_spread_job_ID, 2] == 1,
                #     f"cross_node_{cross_node_spread_job_ID}_indicator_equation")
                #
                # m.addConstr(
                #     (cross_node_sum[cross_node_spread_job_ID, 1] == 1)
                #     >>
                #     (gu.quicksum(cross_node.select(cross_node_spread_job_ID, '*')) >= len(spread_tasks)),
                #     name=f"cross_node_{cross_node_spread_job_ID}_indicator_GREATER_EQUAL_size")
                #
                # m.addConstr(
                #     (cross_node_sum[cross_node_spread_job_ID, 2] == 1)
                #     >>
                #     (gu.quicksum(cross_node.select(cross_node_spread_job_ID, '*')) <= len(spread_tasks) - 1),
                #     name=f"cross_node_{cross_node_spread_job_ID}_indicator_LESS_EQUAL_len-1")

                # cross_node[cross_node_spread_job_ID, node_id] = m.addVar(vtype=GRB.INTEGER,
                #                                                              name=f"{cross_node_spread_job_ID}_on_{node_id}_only_one_indicator")
                # cross_node[cross_node_spread_job_ID, node_id] = gu.quicksum(X[t, a] for t in spread_tasks for a in GPU_IDs_of_node)
                # m.addConstr(
                # cross_node[cross_node_spread_job_ID, node_id] <= len(spread_tasks) - 1,
                # f"cross_node_{cross_node_spread_job_ID}_job_not_on_same_node")

        m.setObjective(gu.quicksum(task_profits[t] * X[t, a] for t in tasks for a in GPUs), GRB.MAXIMIZE)
        m.setParam('TimeLimit', timeout)
        m.update()
        start = time.time_ns()
        m.optimize()
        end = time.time_ns()

        if m.Status != GRB.OPTIMAL:
            info(f"MMKP solver finds unexpected none optimal solution, status = {m.Status}")
            return None

        info(
            f"MMKP solver finds optimal solution, objective value == {m.ObjVal}, duration seconds = {(end - start) / 1e9}")
        assignment: Dict[str, Set[str]] = defaultdict(set)
        for a in GPUs:
            for t in tasks:
                if X[t, a].X < 0.5:
                    continue
                assignment[a].add(t)
        own_calculated_profit = float(cal_profit(task_comp_mem_requirements_and_profits, assignment))
        diff = abs(m.ObjVal - own_calculated_profit)
        info(f"diff with own calculated profit: {diff}")
        if diff > 1e-7:
            info(f"diff > 1e-7: {diff}")

        assigned_tasks = set()
        for tasks in assignment.values():
            assigned_tasks.update(tasks)
        return assignment, cal_profit(task_comp_mem_requirements_and_profits, assignment)


class JobDistributionSolver:
    @staticmethod
    def solve_heuristic(
            partition_to_GPU_IDs: Dict[str, List[str]],
            GPU_comp_mem_total_capacity: Tuple[int, int],
            GPU_comp_mem_capacity: Dict[str, Tuple[int, int]],
            job_comp_mem_demand: Dict[str, Tuple[int, int]],
            job_priority: List[str],
            strategy: str = "heuristic",  # "heuristic" "round"
    ) -> Dict[str, List[str]]:
        partitions = partition_to_GPU_IDs.keys()

        def job_comp(j_):
            return job_comp_mem_demand[j_][0]

        def job_mem(j_):
            return job_comp_mem_demand[j_][1]

        partition_to_jobs = defaultdict(list)

        partition_to_cap = dict()
        for p_, GPU_IDs_ in partition_to_GPU_IDs.items():
            partition_to_cap[p_] = [0, 0]
            for a_ in GPU_IDs_:
                partition_to_cap[p_][0] += GPU_comp_mem_capacity[a_][0]
                partition_to_cap[p_][1] += GPU_comp_mem_capacity[a_][1]

        def heuristic_find_optimal_partition(j_):
            optimal_p = None
            optimal_p_resource_diff = None
            for p__ in partitions:
                remain_comp_cap = partition_to_cap[p__][0] - job_comp(j_)
                remain_mem_cap = partition_to_cap[p__][1] - job_mem(j_)
                if remain_comp_cap < 0 or remain_mem_cap < 0:
                    continue
                remain_comp_cap_norm = 1. * remain_comp_cap / GPU_comp_mem_total_capacity[0]
                remain_mem_cap_norm = 1. * remain_mem_cap / GPU_comp_mem_total_capacity[1]
                resource_diff = abs(remain_comp_cap_norm - remain_mem_cap_norm)
                if (optimal_p is None) or \
                        (resource_diff < optimal_p_resource_diff) or \
                        (resource_diff == optimal_p_resource_diff and len(partition_to_jobs[p__]) < len(
                            partition_to_jobs[optimal_p])):
                    optimal_p = p__
                    optimal_p_resource_diff = resource_diff
            return optimal_p

        def round_find_optimal_partition(j_):
            optimal_p = None
            optimal_p_job_count = None
            for p__ in partitions:
                p_job_count = len(partition_to_jobs[p__])
                remain_comp_cap = partition_to_cap[p__][0] - job_comp(j_)
                remain_mem_cap = partition_to_cap[p__][1] - job_mem(j_)
                if remain_comp_cap < 0 or remain_mem_cap < 0:
                    continue
                if optimal_p_job_count is None or p_job_count < optimal_p_job_count:
                    optimal_p = p__
                    optimal_p_job_count = p_job_count
            return optimal_p

        find_optimal_partition = {
            "heuristic": heuristic_find_optimal_partition,
            "round": round_find_optimal_partition
        }[strategy]

        for job_ in job_priority:
            p_ = find_optimal_partition(job_)
            if p_ is None:
                break
            partition_to_cap[p_][0] -= job_comp(job_)
            partition_to_cap[p_][1] -= job_comp(job_)
            partition_to_jobs[p_].append(job_)

        return partition_to_jobs


def cal_profit(task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, Union[int, float]]],
               assignment: Dict[str, Set[str]]):
    total_profit = 0
    for a in assignment:
        for t in assignment[a]:
            pf = task_comp_mem_requirements_and_profits[t][-1]
            total_profit += pf
    return total_profit


def cal_partition_profit(partition_to_GPU_IDs: Dict[str, List[str]], GPU_ID_to_node_id: Dict[str, str]):
    s = 0
    for p, GPU_IDs in partition_to_GPU_IDs.items():
        node_ids = set(GPU_ID_to_node_id[a] for a in GPU_IDs)
        s += len(node_ids)
    return s


def do_test():
    # test data
    dist_job_to_tasks = {
        "job_ID_148": ("job_ID_148|task_0", "job_ID_148|task_1")
    }
    GPU_to_comp_mem_capacity = {
        "RTX_2080Ti_0": (20, 22),
        "RTX_2080Ti_1": (20, 22),
        "RTX_2080Ti_2": (20, 22),
        "RTX_2080Ti_3": (20, 22)
    }
    task_comp_mem_requirements_and_profits = {
        'job_ID_148|task_0': (10, 11, 2.28181818181818183),
        'job_ID_148|task_1': (10, 11, 2.28181818181818183),
        'job_ID_129|task_0': (4, 5, 0.42727272727272725),
        'job_ID_126|task_0': (4, 4, 0.38181818181818183),
        'job_ID_102|task_0': (4, 5, 0.42727272727272725),
        'job_ID_137|task_0': (12, 17, 1.3727272727272726),
        'job_ID_101|task_0': (4, 5, 0.42727272727272725),
        'job_ID_142|task_0': (6, 7, 0.6181818181818182),
        'job_ID_118|task_0': (6, 6, 0.5727272727272728),
        'job_ID_100|task_0': (16, 13, 1.390909090909091),
        'job_ID_110|task_0': (20, 20, 1.9090909090909092),
        'job_ID_112|task_0': (20, 20, 1.9090909090909092),
        'job_ID_139|task_0': (20, 22, 2.0),
        'job_ID_135|task_0': (14, 20, 1.609090909090909),
        'job_ID_127|task_0': (12, 12, 1.1454545454545455),
        'job_ID_104|task_0': (16, 22, 1.8),
    }

    assignment, duration, profit = AssignmentSolver.MMKP_1(strict=False, dist_job_to_tasks=dist_job_to_tasks,
                                                           GPU_comp_mem_capacity=GPU_to_comp_mem_capacity,
                                                           task_comp_mem_requirements_and_profits=task_comp_mem_requirements_and_profits)
    info(assignment)


def do_test_2():
    # test data
    dist_job_to_tasks = {
        "job_ID_148": ("splitting_true|job_ID_148|task_0", "splitting_true|job_ID_148|task_1")
    }
    GPU_to_comp_mem_capacity = {
        "RTX_2080Ti_0": (20, 22),
        "RTX_2080Ti_1": (20, 22),
    }
    task_comp_mem_requirements_and_profits = {
        'splitting_true|job_ID_148|task_0': (10, 11, 2.28181818181818183),
        'splitting_true|job_ID_148|task_1': (10, 11, 2.28181818181818183),
        'splitting_false|job_ID_148|task_0': (20, 22, 4.28181818181818183),
        # 'job_ID_129|task_0': (4, 5, 0.42727272727272725),
        # 'job_ID_126|task_0': (4, 4, 0.38181818181818183),
    }
    splitting_task_IDs_list_list = [
        ["splitting_true|job_ID_148|task_0", "splitting_true|job_ID_148|task_1", "splitting_false|job_ID_148|task_0"]
    ]
    assignment, profit = AssignmentSolver.MMKP_2(
        splitting_task_IDs_list_list=splitting_task_IDs_list_list,
        dist_job_to_tasks=dist_job_to_tasks,
        GPU_comp_mem_capacity=GPU_to_comp_mem_capacity,
        task_comp_mem_requirements_and_profits=task_comp_mem_requirements_and_profits)
    info(assignment)


def do_test_3():
    # test data
    dist_tasks = [
        ("splitting_2|job_ID_148|task_0", "splitting_2|job_ID_148|task_1"),
        ("splitting_4|job_ID_148|task_0", "splitting_4|job_ID_148|task_1", "splitting_4|job_ID_148|task_2",
         "splitting_4|job_ID_148|task_3"),
        ("splitting_false|job_ID_149|task_0", "splitting_false|job_ID_149|task_1")
    ]

    GPU_to_comp_mem_capacity = {
        "RTX_2080Ti_0": (20, 22),
        "RTX_2080Ti_1": (20, 22),
        "RTX_2080Ti_2": (20, 22),
        "RTX_2080Ti_3": (20, 22),
    }
    task_comp_mem_requirements_and_profits = {
        'splitting_4|job_ID_148|task_0': (5, 5, 1.11181818181818183),
        'splitting_4|job_ID_148|task_1': (5, 5, 1.11181818181818183),
        'splitting_4|job_ID_148|task_2': (5, 5, 1.11181818181818183),
        'splitting_4|job_ID_148|task_3': (5, 5, 1.11181818181818183),
        'splitting_2|job_ID_148|task_0': (10, 11, 2.28181818181818183),
        'splitting_2|job_ID_148|task_1': (10, 11, 2.28181818181818183),
        'splitting_false|job_ID_148|task_0': (20, 22, 4.57181818181818183),
        'splitting_false|job_ID_149|task_0': (10, 11, 5.57181818181818183),
        'splitting_false|job_ID_149|task_1': (10, 11, 5.57181818181818183),
    }
    splitting_job_ID_task_sets = {
        "job_ID_148": [
            ["splitting_false|job_ID_148|task_0"],
            ["splitting_2|job_ID_148|task_1", "splitting_2|job_ID_148|task_0"],
            [
                "splitting_4|job_ID_148|task_0",
                "splitting_4|job_ID_148|task_1",
                "splitting_4|job_ID_148|task_2",
                "splitting_4|job_ID_148|task_3"
            ],
        ]
    }

    assignment, profit = AssignmentSolver.MMKP_3(
        timeout=30,
        splitting_job_ID_task_sets=splitting_job_ID_task_sets,
        dist_tasks=dist_tasks,
        GPU_comp_mem_capacity=GPU_to_comp_mem_capacity,
        task_comp_mem_requirements_and_profits=task_comp_mem_requirements_and_profits)
    info(assignment)


def do_test_4():
    job_ID_to_spread_job_IDs = {
        "job_1": ["job_1|sp_1|in", "job_1|sp_2|in", "job_1|sp_2|cn"],
        "job_2": ["job_2|sp_1|in", "job_2|sp_4|in"],
    }

    spread_job_ID_to_task_sets = {
        "job_1|sp_1|in": ["job_1|sp_1|in|task_0"],
        "job_1|sp_2|in": ["job_1|sp_2|in|task_0", "job_1|sp_2|in|task_1"],
        "job_1|sp_2|cn": ["job_1|sp_2|cn|task_0", "job_1|sp_2|cn|task_1"],
        "job_2|sp_1|in": ["job_2|sp_1|in|task_0"],
        "job_2|sp_4|in": ["job_2|sp_4|in|task_0", "job_2|sp_4|in|task_1", "job_2|sp_4|in|task_2",
                          "job_2|sp_4|in|task_3"]
    }

    GPU_to_comp_mem_capacity = {
        "RTX_2080Ti_0": (3, 22),
        "RTX_2080Ti_1": (5, 22),
        "RTX_2080Ti_2": (5, 22),
        "RTX_2080Ti_3": (5, 22),
    }
    GPU_ID_to_node_id = {
        "RTX_2080Ti_0": "node_0",
        "RTX_2080Ti_1": "node_0",
        "RTX_2080Ti_2": "node_1",
        "RTX_2080Ti_3": "node_1",
    }
    task_comp_mem_requirements_and_profits = {
        'job_1|sp_1|in|task_0': (5, 5, 1),
        'job_1|sp_2|in|task_0': (5, 5, 50),
        'job_1|sp_2|in|task_1': (5, 5, 50),
        'job_1|sp_2|cn|task_0': (1, 1, 10),
        'job_1|sp_2|cn|task_1': (1, 1, 10),
        'job_2|sp_1|in|task_0': (5, 5, 2),
        'job_2|sp_4|in|task_0': (5, 5, 2),
        'job_2|sp_4|in|task_1': (5, 5, 2),
        'job_2|sp_4|in|task_2': (5, 5, 2),
        'job_2|sp_4|in|task_3': (5, 5, 2),
    }

    in_node_spread_job_IDs = ["job_1|sp_1|in",
                              "job_1|sp_2|in",
                              "job_2|sp_1|in",
                              "job_2|sp_4|in"]
    cross_node_spread_job_IDs = ["job_1|sp_2|cn"]

    assignment, profit = AssignmentSolver.MMKP_4(
        timeout=30,
        job_ID_to_spread_job_IDs=job_ID_to_spread_job_IDs,
        spread_job_ID_to_task_sets=spread_job_ID_to_task_sets,
        GPU_comp_mem_capacity=GPU_to_comp_mem_capacity,
        task_comp_mem_requirements_and_profits=task_comp_mem_requirements_and_profits,
        in_node_spread_job_IDs=in_node_spread_job_IDs,
        cross_node_spread_job_IDs=cross_node_spread_job_IDs,
        GPU_ID_to_node_id=GPU_ID_to_node_id)
    info(assignment)


def do_partition_test_1():
    GPU_ID_to_node_id = {
        "GPU_0": "node_0",
        "GPU_1": "node_0",
        "GPU_2": "node_0",
        "GPU_3": "node_0",
        "GPU_4": "node_0",
        "GPU_5": "node_0",
        "GPU_6": "node_0",
        "GPU_7": "node_0",

        "GPU_8": "node_1",
        "GPU_9": "node_1",
        "GPU_10": "node_1",
        "GPU_11": "node_1",
        "GPU_12": "node_1",
        "GPU_13": "node_1",
        "GPU_14": "node_1",
        "GPU_15": "node_1",

        "GPU_16": "node_2",
        "GPU_17": "node_2",
        "GPU_18": "node_2",
        "GPU_19": "node_2",

        "GPU_20": "node_3",
        "GPU_21": "node_3",
        "GPU_22": "node_3",
        "GPU_23": "node_3",
    }
    result = PartitionSolver.solve(GPU_ID_to_node_id=GPU_ID_to_node_id, partition_size=9)
    print(result)
    result = PartitionSolver.solve(GPU_ID_to_node_id=GPU_ID_to_node_id, partition_size=9, strategy="round")
    print(result)


if __name__ == '__main__':
    # do_partition_test_1()
    do_test_4()

import time
from collections import defaultdict
from typing import Dict, Tuple, Union, Optional, Protocol, Set, List

import gurobipy as gu
from gurobipy import GRB

from log import *
from model import SolverResult, SolverParameters, SolverParameters2, SolverParameters3
from object import SolverEnum


def get_solver(solver_enum: 'SolverEnum'):
    return {
        SolverEnum.MMKP: AssignmentSolver.MMKP_1
    }[solver_enum]


def do_solve_1(solver_params: SolverParameters) -> Optional[SolverResult]:
    # info(f"received solver parameters, solver_params: {solver_params}")
    solver = get_solver(SolverEnum[solver_params.solver_type])
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


def do_solve_2(solver_params: SolverParameters2) -> Optional[SolverResult]:
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

def do_solve_3(solver_params: SolverParameters3) -> Optional[SolverResult]:
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


def cal_profit(task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, Union[int, float]]],
               assignment: Dict[str, Set[str]]):
    total_profit = 0
    for a in assignment:
        for t in assignment[a]:
            pf = task_comp_mem_requirements_and_profits[t][-1]
            total_profit += pf
    return total_profit

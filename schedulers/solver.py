import datetime
import json
import os
import time
from collections import defaultdict
from typing import Dict, Tuple, Union, Optional, Protocol, Set

import gurobipy as gu
from gurobipy import GRB

from common import get_session_dir
from log import *
from model import SolverResult, SolverParameters
from object import SolverEnum


def get_solver(solver_enum: 'SolverEnum') -> 'SolverProtocol':
    return {
        SolverEnum.MMKP: AssignmentSolver.MMKP,
        SolverEnum.RoundRobin: AssignmentSolver.round_robin
    }[solver_enum]


class SolverProtocol(Protocol):
    def __call__(self, dist_job_to_tasks: Dict[str, Tuple[str, ...]],
                 GPU_comp_mem_capacity: Dict[str, Tuple[int, int]],
                 task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, Union[int, float]]]) \
            -> Optional[Optional[Tuple[Dict[str, Set[str]], int, Union[int, float]]]]:
        ...


def do_solve(session_id: str,
             solver_params: SolverParameters):
    filename = datetime.datetime.now().strftime(
        f"{session_id}_{solver_params.solver_type.value}_%Y-%m-%d-%H-%M-%S.json")
    session_dir = get_session_dir(session_id)
    if not os.path.exists(session_dir):
        os.mkdir(session_dir)
    filepath = os.path.join(session_dir, filename)
    logging.info(f"received solver parameters, session_id = {session_id}, saving file to {filepath}")
    solver = get_solver(solver_params.solver_type)
    solve_raw_res = solver(dist_job_to_tasks=solver_params.dist_job_to_tasks,
                           GPU_comp_mem_capacity=solver_params.GPU_comp_mem_capacity,
                           task_comp_mem_requirements_and_profits=solver_params.task_comp_mem_requirements_and_profits)
    if solve_raw_res is None:
        return None
    assignment, duration, profit = solve_raw_res
    solver_result = SolverResult(solver_parameters=solver_params, duration=duration, profit=profit,
                                 assignment=assignment)
    with open(filepath, 'w') as f:
        json.dump(solver_result.json(), f)
    return solver_result


class AssignmentSolver:
    @staticmethod
    def MMKP(
            dist_job_to_tasks: Dict[str, Tuple[str, ...]],
            GPU_comp_mem_capacity: Dict[str, Tuple[int, int]],
            task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, Union[int, float]]]) \
            -> Optional[Tuple[Dict[str, Set[str]], int, Union[int, float]]]:
        dist_job_task_size_tasks = {
            dist_job: (len(tasks), tasks) for dist_job, tasks in dist_job_to_tasks.items()
        }
        dist_jobs, dist_job_task_size, dist_job_tasks = gu.multidict(dist_job_task_size_tasks)
        # dist_jobs, dist_job_task_size, dist_job_tasks = gu.multidict({
        #     "job_1": (2, ("task_1_job_1", "task_2_job_1"))
        # })
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

        m.update()
        start = time.time_ns()
        m.optimize()
        end = time.time_ns()

        if m.Status != GRB.OPTIMAL:
            logging.error(f"MMKP solver finds unexpected none optimal solution, status = {m.Status}")
            return None

        logging.info(
            f"MMKP solver finds optimal solution, objective value == {m.ObjVal}, duration seconds = {(end - start) / 1e9}")
        assignment: Dict[str, Set[str]] = defaultdict(set)
        assert m.ObjVal == float(cal_profit(task_comp_mem_requirements_and_profits, assignment))
        for a in GPUs:
            for t in tasks:
                if X[t, a].X < 0.5:
                    continue
                assignment[a].add(t)

        return assignment, end - start, cal_profit(task_comp_mem_requirements_and_profits, assignment)

    @staticmethod
    def round_robin(
            dist_job_to_tasks: Dict[str, Tuple[str, ...]],
            GPU_comp_mem_capacity: Dict[str, Tuple[int, int]],
            task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, Union[int, float]]]) \
            -> Optional[Optional[Tuple[Dict[str, Set[str]], int, Union[int, float]]]]:
        # TODO
        return None


def cal_profit(task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, Union[int, float]]],
               assignment: Dict[str, Set[str]]):
    total_profit = 0
    for a in assignment:
        for t in assignment[a]:
            pf = task_comp_mem_requirements_and_profits[t][-1]
            total_profit += pf
    return total_profit

from typing import Dict, Tuple, Union, Set, Optional

from pydantic import BaseModel

from object import GPUType, SchedulerEnum, SolverEnum


class SolverParameters(BaseModel):
    solver_type: SolverEnum
    GPU_type: GPUType
    dist_job_to_tasks: Dict[str, Tuple[str, ...]]
    GPU_comp_mem_capacity: Dict[str, Tuple[int, int]]
    task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, Union[int, float]]]


class SolverResult(BaseModel):
    solver_parameters: SolverParameters
    duration: int
    profit: Union[float, int]
    assignment: Dict[str, Set[str]]


class SnapshotRecordParameters(BaseModel):
    scheduler_name: str
    scheduler_type: SchedulerEnum
    solver_type: Optional[SolverEnum]
    GPU_type_to_GPU_IDs: Dict[GPUType, Set[str]]
    dist_job_to_tasks: Dict[str, Tuple[str, ...]]
    task_comp_mem_requirements: Dict[str, Tuple[int, int]]
    task_comp_over_supply: Dict[str, int]
    assignments: Dict[str, Set[str]]
    profit: Union[int, float]
    do_plot: bool

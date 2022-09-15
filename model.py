from typing import Dict, Tuple, Union, Set, Optional, List

from pydantic import BaseModel

from object import GPUType, SchedulerEnum, SolverEnum


class SolverParameters(BaseModel):
    class Config:
        use_enum_values = True

    solver_type: SolverEnum
    timeout: int
    strict: bool
    GPU_type: GPUType
    dist_job_to_tasks: Dict[str, Tuple[str, ...]]
    GPU_comp_mem_capacity: Dict[str, Tuple[int, int]]
    task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, float]]


class SolverParameters2(BaseModel):
    class Config:
        use_enum_values = True

    solver_type: SolverEnum
    timeout: int
    splitting_task_IDs_list_list: List[List[str]]
    GPU_type: GPUType
    dist_job_to_tasks: Dict[str, Tuple[str, ...]]
    GPU_comp_mem_capacity: Dict[str, Tuple[int, int]]
    task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, float]]

class SolverParameters3(BaseModel):
    class Config:
        use_enum_values = True

    solver_type: SolverEnum
    timeout: int
    splitting_job_ID_task_sets: Dict[str, List[List[str]]]
    GPU_type: GPUType
    dist_tasks: List[Tuple[str, ...]]
    GPU_comp_mem_capacity: Dict[str, Tuple[int, int]]
    task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, float]]

class SolverResult(BaseModel):
    class Config:
        use_enum_values = True

    solver_parameters: Optional[SolverParameters]
    solver_parameters2: Optional[SolverParameters2]
    solver_parameters3: Optional[SolverParameters3]
    duration: int
    profit: int
    assignment: Dict[str, Set[str]]


class SnapshotRecordParameters(BaseModel):
    class Config:
        use_enum_values = True
    scheduler_name: str
    scheduler_type: SchedulerEnum
    solver_type: Optional[SolverEnum]
    GPU_type_to_GPU_IDs: Dict[GPUType, Set[str]]
    dist_job_to_tasks: Dict[str, Tuple[str, ...]]
    task_comp_mem_requirements: Dict[str, Tuple[int, int]]
    task_comp_over_supply: Dict[str, int]
    task_comp_lack_supply: Dict[str, int]
    assignments: Dict[str, Set[str]]
    profit: float
    do_plot: Optional[bool]

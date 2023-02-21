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


class SolverParameters4(BaseModel):
    class Config:
        use_enum_values = True

    solver_type: SolverEnum
    timeout: int
    job_ID_to_spread_job_IDs: Dict[str, List[str]]
    spread_job_ID_to_task_sets: Dict[str, List[str]]
    GPU_comp_mem_capacity: Dict[str, Tuple[int, int]]
    in_node_spread_job_IDs: List[str]
    cross_node_spread_job_IDs: List[str]
    dist_tasks: List[Tuple[str, ...]]
    task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, float]]
    GPU_ID_to_node_id: Dict[str, str]

class SolverResult(BaseModel):
    class Config:
        use_enum_values = True

    solver_parameters: Optional[SolverParameters]
    solver_parameters2: Optional[SolverParameters2]
    solver_parameters3: Optional[SolverParameters3]
    solver_parameters4: Optional[SolverParameters4]
    duration: int
    profit: int
    assignment: Dict[str, Set[str]]


class PartitionSolverParameters(BaseModel):
    class Config:
        use_enum_values = True

    timeout: int
    GPU_ID_to_node_id: Dict[str, str]
    partition_size: int
    strategy: str


class PartitionSolverResult(BaseModel):
    class Config:
        use_enum_values = True

    solver_parameters: Optional[PartitionSolverParameters]
    duration: int
    GPU_ID_to_partition: Dict[str, str]
    partition_to_GPU_IDs: Dict[str, List[str]]
    partition_profit: int


class JobDistributionSolverParameters(BaseModel):
    class Config:
        use_enum_values = True

    partition_to_GPU_IDs: Dict[str, List[str]]
    GPU_comp_mem_capacity: Dict[str, Tuple[int, int]]
    GPU_comp_mem_total_capacity: Tuple[int, int]
    job_comp_mem_demand: Dict[str, Tuple[int, int]]
    job_priority: List[str]
    strategy: str


class JobDistributionSolverResult(BaseModel):
    class Config:
        use_enum_values = True

    solver_parameters: Optional[JobDistributionSolverParameters]
    partition_to_jobs: Dict[str, List[str]]

class SnapshotRecordParameters(BaseModel):
    class Config:
        use_enum_values = True
    now: int
    scheduler_name: str
    scheduler_type: SchedulerEnum
    solver_type: Optional[SolverEnum]
    waiting_jobs: int
    done_jobs: int
    running_jobs: int
    dist_jobs: int
    spread_jobs: int
    GPU_type_to_GPU_IDs: Dict[GPUType, Set[str]]
    dist_job_to_tasks: Dict[str, Tuple[str, ...]]
    task_comp_mem_requirements: Dict[str, Tuple[int, int]]
    task_comp_over_supply: Dict[str, int]
    task_comp_lack_supply: Dict[str, int]
    assignments: Dict[str, Set[str]]
    profit: float
    do_plot: Optional[bool]

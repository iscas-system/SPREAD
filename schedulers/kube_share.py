from collections import defaultdict
from typing import Tuple, Optional, Set, Dict, List, Any

from cluster import TaskAssignment, Assignments
from object import CompCapacity, GPUType, Task
from scheduler import Scheduler
from schedulers.sorter import Sorter


class KubeShareScheduler(Scheduler):
    def _init_config(self):
        ...

    def do_assign(self, preemptive: bool) -> Tuple[Assignments, Optional[Any]]:
        jobs = self.cluster.jobs
        job_IDs = Sorter.sort(jobs=jobs.values(), data_source=self.data_source, priority_type=self.priority_type)
        curr_GPU_ID_idx = -1
        GPU_IDs = sorted(self.cluster.GPU_IDs)
        if not preemptive:
            GPU_ID_to_task_assignments = self.cluster.assignments.GPU_ID_to_task_assignments
        else:
            GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]] = defaultdict(set)

        return ..., None
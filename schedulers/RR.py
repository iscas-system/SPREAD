from typing import Tuple, Optional

from scheduler import Scheduler
from sorter import Sorter
from object import PriorityType


class RR(Scheduler):
    def _init_config(self):
        ...

    def do_assign(self, preemptive: bool) -> Tuple[Optional[Tuple[int, ...]],]:
        jobs = self.cluster.jobs.values()
        job_IDs = Sorter.sort(jobs=jobs, data_source=self.data_source, priority_type=self.priority_type)
        curr_GPU_ID_idx = 0

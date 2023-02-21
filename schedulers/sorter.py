from typing import Iterable, List, Dict
from object import PriorityType, Job
from data_source import DataSource
from functools import cmp_to_key


class Sorter:
    @staticmethod
    def sort(jobs: Iterable[Job], data_source: DataSource, priority_type: PriorityType) -> List[str]:
        return {
            PriorityType.SRSF: Sorter.SRSF,
            PriorityType.FCFS: Sorter.FCFS
        }[priority_type](jobs, data_source)

    @staticmethod
    def SRSF(jobs: Iterable[Job], data_source: DataSource) -> List[str]:
        jobs = list(jobs)
        job_remaining_services: Dict[str, int] = dict()
        for job in jobs:
            job_spec = data_source.get_job_spec(job_ID=job.job_ID)
            remain_ratio = job.remaining_iterations / job_spec.total_iterations
            remain_time = remain_ratio * job_spec.run_time_nano
            job_remaining_services[job.job_ID] = int(job_spec.plan_GPU * remain_time)

        jobs.sort(key=lambda job_ID: job_remaining_services[job_ID])
        job_IDs = [job.job_ID for job in jobs]
        return job_IDs

    @staticmethod
    def FCFS(jobs: Iterable[Job], data_source: DataSource) -> List[str]:
        job_IDs = [job.job_ID for job in jobs]
        def cmp(job_ID_1: str, job_ID_2: str):
            job_spec_1 = data_source.get_job_spec(job_ID=job_ID_1)
            job_spec_2 = data_source.get_job_spec(job_ID=job_ID_2)
            if job_spec_1.submit_time_nano == job_spec_2.submit_time_nano:
                return 1 if job_spec_1.job_ID < job_spec_2.job_ID else -1
            return 1 if job_spec_1.submit_time_nano < job_spec_2.submit_time_nano else -1

        job_IDs.sort(key=cmp_to_key(cmp))
        return job_IDs

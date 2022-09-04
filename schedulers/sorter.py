from typing import Iterable, List, Dict
from object import PriorityType, Job
from data_source import DataSource


class Sorter:
    @staticmethod
    def sort(jobs: Iterable[Job], data_source: DataSource, priority_type: PriorityType) -> List[str]:
        return {
            PriorityType.SRSF: Sorter.SRSF
        }[priority_type](jobs, data_source)

    @staticmethod
    def SRSF(jobs: Iterable[Job], data_source: DataSource) -> List[str]:
        jobs = list(jobs)
        job_remaining_services: Dict[str, int] = dict()
        for job in jobs:
            job_spec = data_source.get_job_spec(job_ID=job.job_ID)
            remain_ratio = job.remaining_iterations / job_spec.total_iterations
            remain_time = remain_ratio * job_spec.run_time
            job_remaining_services[job.job_ID] = int(job_spec.plan_GPU * remain_time)

        jobs.sort(key=lambda job_ID: job_remaining_services[job_ID])
        job_IDs = [job.job_ID for job in jobs]
        return job_IDs

    @staticmethod
    def FCFS(jobs: Iterable[Job], data_source: DataSource) -> List[str]:
        job_IDs = [job.job_ID for job in jobs]
        job_IDs.sort(key=lambda job_ID: data_source.get_job_spec(job_ID=job_ID).submit_time)
        return job_IDs

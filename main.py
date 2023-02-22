import argparse
import os
import pathlib
import datetime
from itertools import product

from config import get_config, Config
from log import set_logging
from simulator import Simulator
from multiprocessing import Process

def main():
    parser = argparse.ArgumentParser(description='MMKP')
    parser.add_argument('--config-path',
                        type=str,
                        required=False,
                        default=str(pathlib.Path(__file__).parent / "configs" / "MMKP_config.json"),
                        help="config path")
    args = parser.parse_args()
    assert os.path.exists(args.config_path), "config path not exists"
    run(args.config_path)


def run(config_path: str):
    c = get_config(config_path)
    if c.multiprocessing:
        processes = list()
        for data_source_config_name, cluster_config_name, scheduler_name in product(c.enabled_data_source_configs,
                                                                    c.enabled_cluster_configs,
                                                                    c.enabled_scheduler_names):
            p = Process(target=simulator_process, args=(config_path, data_source_config_name, cluster_config_name, scheduler_name))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        for data_source_config_name, cluster_config_name, scheduler_name in product(c.enabled_data_source_configs,
                                                                    c.enabled_cluster_configs,
                                                                    c.enabled_scheduler_names):
            simulator_process(config_path, data_source_config_name, cluster_config_name, scheduler_name)

def simulator_process(config_path: str, data_source_config_name: str, cluster_config_name: str, scheduler_name: str):
    c = get_config(config_path)
    time_str = datetime.datetime.now().strftime(
        f"%Y-%m-%d-%H-%M-%S")
    set_logging(f"Player_{c.session_id}_{cluster_config_name}_{data_source_config_name}_{scheduler_name}_{time_str}.log")
    data_source_config = c.data_source_configs[data_source_config_name]
    cluster_config = c.cluster_configs[cluster_config_name]
    sim = Simulator(data_source_config=data_source_config, cluster_config=cluster_config, scheduler_name=scheduler_name)
    sim.play()


if __name__ == '__main__':
    main()
    # do_test_3()

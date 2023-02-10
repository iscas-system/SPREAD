import argparse
import os
import pathlib
from itertools import product

from config import get_config, Config
from log import info
from schedulers.solver import AssignmentSolver
from simulator import Simulator

def main():
    parser = argparse.ArgumentParser(description='MMKP')
    parser.add_argument('--config-path',
                        type=str,
                        required=False,
                        default=str(pathlib.Path(__file__).parent / "configs" / "test_config.json"),
                        help="config path")
    args = parser.parse_args()
    assert os.path.exists(args.config_path), "config path not exists"
    c = get_config(args.config_path)
    run(c)


def run(c: Config):
    for data_source_config_name, cluster_config_name in product(c.enabled_data_source_configs,
                                                                c.enabled_cluster_configs):
        data_source_config = c.data_source_configs[data_source_config_name]
        cluster_config = c.cluster_configs[cluster_config_name]
        sim = Simulator(data_source_config=data_source_config, cluster_config=cluster_config)
        sim.play()


if __name__ == '__main__':
    main()
    # do_test_3()

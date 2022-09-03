import logging
import sys


def init_logging():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')

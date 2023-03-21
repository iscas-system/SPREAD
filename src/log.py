import logging
import os.path
import sys


def set_logging(to_stdout: bool, file_name: str):
    if to_stdout:
        logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    else:
        logging.basicConfig(filename=os.path.join("logs", file_name),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

def info(msg: str):
    # print(f"[{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}] {msg}")
    logging.info(msg)

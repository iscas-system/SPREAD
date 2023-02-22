import logging
import os.path

def set_logging(file_name: str):
    logging.basicConfig(filename=os.path.join("logs", file_name),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

def info(msg: str):
    # print(f"[{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}] {msg}")
    logging.info(msg)

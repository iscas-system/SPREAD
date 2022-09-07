import datetime


def info(msg: str):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}] {msg}")

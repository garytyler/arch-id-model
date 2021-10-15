from functools import wraps
from time import time
from typing import Callable


def func_timer(f: Callable):
    @wraps(f)
    def wrap(*args, **kwargs):
        time_start = time()
        print(f"Starting: func={f.__name__}")
        result = f(*args, **kwargs)
        time_end = time()
        print(
            ", ".join(
                (
                    f"Finished: func={f.__name__}",
                    f"args={args}",
                    f"kwargs={kwargs}",
                    f"took={time_end - time_start:2.4f} sec",
                )
            )
        )
        return result

    return wrap

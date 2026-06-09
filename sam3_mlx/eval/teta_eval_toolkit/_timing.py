"""Timing decorator retained from the official toolkit shape."""

from __future__ import annotations

import inspect
from functools import wraps
from time import perf_counter

DO_TIMING = False
DISPLAY_LESS_PROGRESS = False
timer_dict = {}
counter = 0


def time(f):
    @wraps(f)
    def wrap(*args, **kw):
        if not DO_TIMING:
            return f(*args, **kw)
        ts = perf_counter()
        result = f(*args, **kw)
        te = perf_counter()
        tt = te - ts
        arg_names = inspect.getfullargspec(f)[0]
        if arg_names and arg_names[0] == "self" and DISPLAY_LESS_PROGRESS:
            return result
        method_name = (
            type(args[0]).__name__ + "." + f.__name__
            if arg_names and arg_names[0] == "self"
            else f.__name__
        )
        timer_dict[method_name] = timer_dict.get(method_name, 0) + tt
        if method_name == "Evaluator.evaluate":
            print("")
            print("Timing analysis:")
            for key, value in timer_dict.items():
                print("%-70s %2.4f sec" % (key, value))
        return result

    return wrap

from functools import reduce as call_reduce

from utils.functions import safe_call

def reduce(array, callback, default_value=None):
    reducer = lambda accumulator, item: safe_call(callback, accumulator, item[1], item[0])

    return call_reduce(reducer, enumerate(array), default_value)

def map(array, callback):
    mapper = lambda item: safe_call(callback, item[1], item[0])

    return [mapper(item) for item in enumerate(array)]

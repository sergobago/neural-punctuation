from utils.functions import safe_function

def some(array, callable_condition):
    callback = safe_function(callable_condition)

    for index, value in enumerate(array):
        if callback(value, index, array):
            return True

    return False

def every(array, callable_condition):
    callback = safe_function(callable_condition)

    for index, value in enumerate(array):
        if not callback(value, index, array):
            return False

    return True

def find_index(array, callable_condition, default_value=None):
    callback = safe_function(callable_condition)

    for index, value in enumerate(array):
        if callback(value, index, array):
            return index

    return default_value

def find_rindex(array, callable_condition, default_value=None):
    callback = safe_function(callable_condition)

    for index, value in enumerate(reversed(array)):
        if callback(value, index, array):
            return len(array) - index - 1

    return default_value

def find(array, callable_condition, default_value=None):
    callback = safe_function(callable_condition)

    for index, value in enumerate(array):
        if callback(value, index, array):
            return value

    return default_value

def filter(array, callable_condition):
    new_array = []
    callback = safe_function(callable_condition)

    for index, value in enumerate(array):
        if callback(value, index, array):
            new_array.append(value)

    return new_array

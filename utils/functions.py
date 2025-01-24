from inspect import isfunction, ismethod

def is_function(func):
    return isfunction(func)

def is_method(func):
    return ismethod(func)

def is_callable(func):
    return callable(func)

def get_function_name(func):
    return func.__name__

def get_function_all_parameter_count(func):
    return func.__code__.co_argcount

def get_function_arg_count(func):
    return get_function_all_parameter_count(func) - get_function_kwarg_count(func)

def get_function_kwarg_count(func):
    return len(get_function_kwarg_values(func))

def get_function_kwarg_values(func):
    return func.__defaults__ or tuple()

def get_function_all_parameter_names(func):
    return set(func.__code__.co_varnames)

def get_function_allowed_args(func, *args):
    if not args:
        return args

    arg_count = get_function_arg_count(func)

    if is_method(func):
        arg_count = max(0, arg_count - 1)

    return args[:arg_count]

def get_function_allowed_kwargs(func, **kwargs):
    if not kwargs:
        return kwargs

    all_parameter_names = get_function_all_parameter_names(func)

    return { key:kwargs[key] for key in kwargs if key in all_parameter_names }

def safe_call(func, *args, **kwargs):
    new_args = get_function_allowed_args(func, *args)
    new_kwargs = get_function_allowed_kwargs(func, **kwargs)

    return func(*new_args, **new_kwargs)

def safe_function(func):
    return lambda *args, **kwargs: safe_call(func, *args, **kwargs)

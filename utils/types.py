def to_int(value, default_value=0):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default_value

def to_float(value, default_value=0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default_value

def to_str(value, default_value=''):
    if value is None:
        return default_value

    try:
        return str(value)
    except (ValueError, TypeError):
        return default_value

def to_boolean(value):
    return bool(value)

def is_int(value):
    if is_boolean(value):
        return False

    if is_float(value):
        return value.is_integer()

    return isinstance(value, int)

def is_float(value):
    return isinstance(value, float)

def is_number(value):
    return is_float(value) or is_int(value)

def is_str(value):
    return isinstance(value, str)

def is_boolean(value):
    return isinstance(value, bool)

def is_numeric(value):
    if is_boolean(value):
        return False

    numeric_value = to_float(value, default_value=None)

    return numeric_value is not None

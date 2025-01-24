from math import ceil
from numpy import ndarray
from random import choice

from utils.loops import reduce

def is_array(value):
    return isinstance(value, (list, tuple, set, ndarray))

def is_dictionary(value):
    return isinstance(value, dict)

def get_normalized_array(value, included_value=False):
    if is_array(value):
        return list(value)

    return [value] if included_value else []

def get_normalized_dictionary(value):
    return dict(value) if is_dictionary(value) else dict()

def get_index(*args, **kwargs):
    return get_key(*args, **kwargs)

def get_key(collection, key, default_value=None):
    try:
        return collection[key]
    except (IndexError, KeyError, TypeError):
        return default_value

def check_has_key(collection, key):
    try:
        return bool(collection[key]) or True
    except (IndexError, KeyError, TypeError):
        return False

def difference(array1, array2):
    return list(set(array1) - set(array2))

def get_first_index(array, default_value=None):
    return get_index(array, 0, default_value)

def fill(number_copies, item):
    return [item] * number_copies

def padding_array(array, length, padding_value=0, to_beginning=False):
    if len(array) > length - 1:
        return array[:length]

    zeros = fill(length - len(array), padding_value)

    return zeros + array if to_beginning else array + zeros

def find_indexes(array, value):
    return [index for index, val in enumerate(array) if val == value]

def find_index(array, value, default_value=None, start=0, end=None):
    try:
        end_pos = len(array) if end is None else end

        return array.index(value, start, end_pos)
    except ValueError:
        return default_value

def find_rindex(array, value, default_value=None, start=0, end=None):
    try:
        end_pos = len(array) if end is None else end

        return array.rindex(value, start, end_pos)
    except ValueError:
        return default_value

def multi_extend(array_with_arrays):
    result = []

    for array in array_with_arrays:
        result.extend(array)

    return result

def get_array_step_parts(array, step):
    size = len(array)

    return [array[i:i + step] for i in range(0, size, step)]

def get_array_parts(array, number_parts):
    step = ceil(len(array) / number_parts)

    return get_array_step_parts(array, step)

def get_array_part(array, number_parts, part_position):
    array_length = len(array)
    step = ceil(array_length / number_parts)
    start_position = (part_position - 1) * step

    return array[start_position:start_position + step]

def get_list_keys(dictionary):
    return list(dictionary.keys())

def reverse(array):
    return array[::-1]

def random_choice(array):
    return choice(array) if array else None

def closure_order_choice(initial_index=0):
    next_index = initial_index

    def order_choice(array):
        nonlocal next_index
        current_index = next_index
        next_index = next_index + 1 if len(array) > next_index + 1 else 0

        return get_index(array, current_index)

    return order_choice

def get_dict_first_value(dictionary, default_value=None):
    return get_first_index(dictionary.values(), default_value) if dictionary else None

def invert_dictionary(dictionary):
    return {value: key for key, value in dictionary.items()}

def sort_dict_values(array, reverse=False):
    return dict(sorted(array.items(), key=lambda item: item[1], reverse=reverse))

def truncate_dict(array, max_count_items):
    return {key: value for key, value in list(array.items())[:max_count_items]}

def pop(array, key, default_value=None):
    try:
        return array.pop(key)
    except (IndexError, KeyError):
        return default_value

def has_value_or_zero(value):
    return value or value == 0

def is_empty(value):
    if is_array(value):
        return is_empty_array(value)

    if is_dictionary(value):
        return is_empty_dictionary(value)

    return not has_value_or_zero(value)

def is_empty_array(value):
    return not value

def is_empty_dictionary(value):
    return not value

def join_parameters(parameters, separator=' '):
    return separator.join([str(parameter) for parameter in parameters if has_value_or_zero(parameter)])

def join_by_separators(parameters, separators):
    new_parameters = []

    for parameter_index, parameter in enumerate(parameters):
        separator = separators[parameter_index]
        new_parameter = parameter + separator if separator else parameter
        new_parameters.append(new_parameter)

    return ''.join(new_parameters)

def get_array_indexes(array):
    return list(range(len(array))) if array else []

def set_property(collection, keys, value):
    key = get_first_index(keys)
    rest_keys = keys[1:]
    is_last_key = len(rest_keys) == 0
    has_current_value = check_has_key(collection, key)

    if is_last_key and is_array(collection) and not has_current_value:
        collection.insert(key, value)

        return

    if is_last_key:
        collection[key] = value

        return

    if not has_current_value:
        next_key = get_first_index(rest_keys)
        is_key_integer = isinstance(next_key, int) and not isinstance(next_key, bool)
        collection[key] = [] if is_key_integer else dict()

    return set_property(collection[key], rest_keys, value)

def get_property(accumulator, key):
    return get_key(accumulator, key) if accumulator else None

def get_length(value):
    return len(value) if has_value_or_zero(value) else 0

def safeget(dictionary, keys, default_value=None):
    if not keys:
        return default_value

    return reduce(keys, get_property, dictionary) or default_value

def concatenate_arrays(arrays):
    return reduce(arrays, lambda accumulator, current_array: accumulator + current_array, [])

def flatten_array(array):
    values = []

    for item in array:
        new_values = flatten_array(item) if is_array(item) else [item]
        values.extend(new_values)

    return values

def flatten_dictionary(value, keys=None):
    values = dict()
    parent_keys = get_normalized_array(keys)

    if is_array(value):
        for index, item in enumerate(value):
            new_values = flatten_dictionary(item, keys=[*parent_keys, index])
            values.update(new_values)

    if is_dictionary(value):
        for key, item in value.items():
            new_values = flatten_dictionary(item, keys=[*parent_keys, key])
            values.update(new_values)

    if keys and not values:
        values[tuple(keys)] = value

    return values

def is_equal(value1, value2):
    if is_dictionary(value1) and is_dictionary(value2):
        return is_dictionary_equal(value1, value2)

    if is_array(value1) and is_array(value2):
        return is_array_equal(value1, value2)

    return value1 == value2

def is_dictionary_equal(dictionary1, dictionary2):
    keys1 = get_list_keys(dictionary1)
    keys2 = get_list_keys(dictionary2)

    if len(keys1) != len(keys2):
        return False

    comparisons = [key in keys2 and is_equal(dictionary1.get(key), dictionary2.get(key)) for key in keys1]

    return all(comparisons)

def is_array_equal(array1, array2):
    if len(array1) != len(array2):
        return False

    comparisons = [is_equal(value1, value2) for value1, value2 in zip(array1, array2)]

    return all(comparisons)

def get_overflow_index(array, index):
    return index % len(array)

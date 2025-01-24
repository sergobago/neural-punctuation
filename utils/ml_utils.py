from os import listdir

from utils.array import find_index, get_index
from utils.types import to_float, to_int

def get_checkpoint_parameter(checkpoint_name, parameter_name):
    parameters = checkpoint_name.split('=') if checkpoint_name else None
    parameter_index = find_index(parameters, parameter_name) if parameters else None
    parameter_value = get_index(parameters, parameter_index + 1) if parameter_index is not None else None

    return parameter_value

def get_last_checkpoint(path_dir, metric_name='val_loss', is_need_max=False):
    best_model = None
    current_metric_value = None

    for model_name in listdir(path_dir):
        is_best_metric = False
        parameter_value = get_checkpoint_parameter(model_name, metric_name)
        metric_value = to_float(parameter_value)

        if current_metric_value is not None:
            is_best_metric = current_metric_value < metric_value if is_need_max else current_metric_value > metric_value

        if is_best_metric or best_model is None:
            current_metric_value = metric_value
            best_model = model_name

    return best_model

def get_checkpoint_epoch(model_name):
    parameter_value = get_checkpoint_parameter(model_name, 'epoch')

    return to_int(parameter_value) if parameter_value else 0

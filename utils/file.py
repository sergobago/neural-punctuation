from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from os import listdir, unlink
from os.path import getmtime, isfile, islink, join
from shutil import rmtree

from utils.exists import is_file_exist
from utils.json_parse import json_format_dumps, json_loads
from utils.path import get_dir_numbered_paths

def sum_size_files(files):
    return sum([len(file) for file in files]) if files else None

def read_files(files, max_size=None, max_items=None):
    read_files = []
    max_file_size = max_size or -1

    if not files:
        return None

    for index, file in enumerate(files):
        if max_items and index + 1 > max_items:
            break

        file_data = file.read(max_file_size)
        file.close()
        read_files.append(file_data)

    return read_files

def clear_dir(dir_path, age_hours=0):
    for filename in listdir(dir_path):
        file_path = join(dir_path, filename)
        file_modified = datetime.fromtimestamp(getmtime(file_path))

        if datetime.now() - file_modified <= timedelta(hours=age_hours):
            continue

        remove_file(file_path)

def remove_file(file_path):
    if isfile(file_path) or islink(file_path):
        unlink(file_path)
    else:
        rmtree(file_path)

def load_numbered_files(dir_path, mode='rb', reverse=False):
    file_paths = get_dir_numbered_paths(dir_path, reverse=reverse)

    return load_files(file_paths, mode)

def load_files(file_paths, mode='rb'):
    files = []

    for file_path in file_paths:
        file_data = load_file(file_path, mode)

        if file_data is None:
            continue

        files.append(file_data)

    return files

def file_write(file, text, from_new_line=False):
    file.write(text)

    if from_new_line:
        file.write('\n')

def load_file(file_path, mode='r', encoding=None, is_readlines=False, is_json=False):
    if not is_file_exist(file_path):
        return None

    with open(file_path, mode=mode, encoding=encoding) as file:
        if is_json:
            return json_loads(file.read())

        if is_readlines:
            return file.readlines()

        return file.read()

def save_binary_file(file_path, file_data):
    if not file_path or not file_data:
        return None

    with open(file_path, 'wb') as file:
        file.write(file_data)

    return file_path

def save_binary_files(dir_path, files_data, extension_getter=None):
    input_paths = []

    for file_data_index, file_data in enumerate(files_data):
        file_extension = extension_getter(file_data, file_data_index) if extension_getter else None
        filename = f'{file_data_index}.{file_extension}' if file_extension else str(file_data_index)
        input_path = join(dir_path, filename)
        input_paths.append(input_path)

    with ThreadPoolExecutor() as executor:
        output_paths = list(executor.map(save_binary_file, input_paths, files_data))

    return [output_path for output_path in output_paths if output_path]

def save_file(file_path, value=None, mode='w', encoding=None, formatter=None):
    with open(file_path, mode, encoding=encoding) as file:
        formatted_value = formatter(value) if formatter else value
        file.write(formatted_value)

def save_json_file(*args, **kwargs):
    save_file(*args, formatter=json_format_dumps, **kwargs)

def get_file_number_lines(file_path, encoding=None):
    with open(file_path, 'r', encoding=encoding) as file:
        return sum(1 for _ in file)

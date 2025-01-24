from os import listdir, makedirs
from os.path import basename, isdir, join, splitext

from utils.array import get_first_index, get_index
from utils.types import to_int

def get_path_filename(path, new_extension=None, with_extension=True, prefix=None):
    extension = new_extension if new_extension else get_path_extension(path)
    filename = get_first_index(splitext(basename(path)))

    if prefix:
        filename = f'{filename}_{prefix}'

    if not with_extension:
        return filename

    return f'{filename}.{extension}'

def get_path_extension(path):
    extension = get_index(splitext(path), -1)

    return extension[1:] if extension else None

def get_dir_numbered_paths(dir_path, reverse=False):
    filenames = get_numbered_filenames(dir_path, reverse=reverse)

    return [join(dir_path, filename) for filename in filenames]

def get_numbered_filenames(dir_path, reverse=False):
    if not isdir(dir_path):
        return None

    return sorted(listdir(dir_path), key=lambda filename: to_int(get_path_filename(filename, with_extension=False)), reverse=reverse)

def join_url_paths(*params):
    return '/'.join(params)

def safe_make_dir(dir_path):
    makedirs(dir_path, exist_ok=True)

    return dir_path

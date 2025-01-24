from os.path import exists, isdir

def is_exist(path):
    return path and exists(path)

def is_directory_exist(dir_path):
    return is_exist(dir_path) and isdir(dir_path)

def is_file_exist(file_path):
    return is_exist(file_path) and not isdir(file_path)

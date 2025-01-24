from filetype import guess_extension

def get_file_extension(file_data):
    return guess_extension(file_data) if file_data else None

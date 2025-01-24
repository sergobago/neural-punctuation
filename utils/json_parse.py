from json import dump, dumps, load, loads

def json_loads(value):
    try:
        return loads(value)
    except ValueError:
        return None

def json_dumps(value, default=str, ensure_ascii=False, **kwargs):
    return dumps(value, default=default, ensure_ascii=ensure_ascii, **kwargs)

def json_format_dumps(value, indent=2, **kwargs):
    return json_dumps(value, indent=indent, *kwargs)

def json_write_file(file, value, default=str, ensure_ascii=False, **kwargs):
    return dump(value, file, ensure_ascii=ensure_ascii, default=default, **kwargs)

def json_load_file(file):
    return load(file)

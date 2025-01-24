def insert_string_to_position(array, index, value):
    return array[:index] + value + array[index:]

def get_unique_letters(text):
    return ''.join(set(list(text)))

def get_strip_parameters(value, left_strip=True, right_strip=True):
    text = value
    prefix = ''
    postfix = ''

    if left_strip and text:
        new_text = text.lstrip()
        prefix = text[:len(text) - len(new_text)]
        text = new_text

    if right_strip and text:
        new_text = text.rstrip()
        postfix = text[len(new_text):]
        text = new_text

    return dict(text=text, prefix=prefix, postfix=postfix)

def get_striped(text):
    return text.strip() if text else text

def get_striped_left(text):
    return text.lstrip() if text else text

def get_striped_right(text):
    return text.rstrip() if text else text

def is_alphabet(value):
    return value.isalpha()

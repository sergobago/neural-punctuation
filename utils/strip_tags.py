from re import compile, sub

_HTML_TAG_REGEX = compile(r'<[^<>]*>+')
_SPECIAL_CHARACTER_REGEX = compile(r'&+\S*;+')
_LONG_SPACE_REGEX = compile(r'\s+')

def strip_tags(text, repl=''):
    return sub(_HTML_TAG_REGEX, repl, text) if text else text

def strip_special_characters(text, repl=''):
    return sub(_SPECIAL_CHARACTER_REGEX, repl, text.replace('&nbsp;', ' ')) if text else text

def normalize_spaces(text, repl=' '):
    return sub(_LONG_SPACE_REGEX, repl, text.replace(' ', ' ')) if text else text

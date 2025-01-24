from utils.string import get_unique_letters

from constants.languages import DE, EN, ES, FR, KZ, RU, UA

ALPHABETS = {
    RU: 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
    EN: 'abcdefghijklmnopqrstuvwxyz',
    FR: 'aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz',
    DE: 'aäbcdefghijklmnoöpqrstuüvwxyz',
    KZ: 'аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя',
    UA: 'абвгґдеєжзиіїйклмнопрстуфхцчшщьюя',
    ES: 'abcdefghijklmnñopqrstuvwxyz',
}

DOT_LETTER = '.'
ELLIPSIS_LETTER = '…'
EXCLAMATION_LETTER = '!'
QUESTION_MARK = '?'
COMMA_LETTER = ','
COLON_LETTER = ':'
SEMICOLON_LETTER = ';'
END_SENTENCE = DOT_LETTER + EXCLAMATION_LETTER + QUESTION_MARK + ELLIPSIS_LETTER
BASE_PUNCTUATIONS = COMMA_LETTER + COLON_LETTER + SEMICOLON_LETTER
ALL_PUNCTUATIONS = END_SENTENCE + BASE_PUNCTUATIONS
DASHES = '-‑−–—―_'
DIGITS = '1234567890'
QUOTES = '\'„”“"’`'
LEFT_BRACKETS = '({[«<'
RIGHT_BRACKETS = ')}]»>'
CURRENCIES = '%$₽£€'
SLASHES = '/|\\'
MATHS = '*+=~^'
TAGS = '#№§'
SYMBOLS = '&π'
SPACE = ' '
UNIQUE_BASE_CHARS = get_unique_letters(ALL_PUNCTUATIONS + SYMBOLS + MATHS + TAGS + QUOTES + LEFT_BRACKETS + RIGHT_BRACKETS + CURRENCIES + SLASHES + DASHES)
UNIQUE_ALPHABET_LETTERS = get_unique_letters(''.join(ALPHABETS.values()))
UNIQUE_ALL_LETTERS = SPACE + UNIQUE_BASE_CHARS + UNIQUE_ALPHABET_LETTERS + DIGITS

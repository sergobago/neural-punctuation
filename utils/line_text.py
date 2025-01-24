from re import DOTALL, compile, escape, search, sub

from utils.array import get_index, pop
from utils.helpers import find_index, find_rindex
from utils.strip_tags import normalize_spaces

from constants.letters import (ALL_PUNCTUATIONS, BASE_PUNCTUATIONS, COLON_LETTER, COMMA_LETTER, CURRENCIES, DASHES,
                               DIGITS, DOT_LETTER, ELLIPSIS_LETTER, END_SENTENCE, EXCLAMATION_LETTER, LEFT_BRACKETS,
                               MATHS, QUESTION_MARK, QUOTES, RIGHT_BRACKETS, SEMICOLON_LETTER, SLASHES, SPACE, SYMBOLS,
                               TAGS, UNIQUE_ALL_LETTERS, UNIQUE_ALPHABET_LETTERS, UNIQUE_BASE_CHARS)

_PROTECTED_REPLACERS = dict(NUMBER='1', MAIL='2', LINK='3')
_ENTERS_REGEX = compile(r'\r|\n|\t|\\r|\\n|\\t', DOTALL)
_NUMBERS_REGEX = compile(rf'\d+([{escape(ALL_PUNCTUATIONS + MATHS + DASHES + SLASHES)}]+\d+)*', DOTALL)
_STANDARDIZATION_URLS_REGEX = compile(rf'(\S*[{escape(SLASHES + COLON_LETTER + DOT_LETTER)}]){{2}}\S*(?<![{escape(ALL_PUNCTUATIONS)}])', DOTALL)
_STANDARDIZATION_DOGS_REGEX = compile(rf'\S*\@+\S*(?<![{escape(ALL_PUNCTUATIONS)}])', DOTALL)
_STANDARDIZATION_NUMBER_WITH_BRACKETS_REGEX = compile(rf'[{escape(LEFT_BRACKETS[0])}]+[\s\d]*\S?[\s\d]*[{escape(RIGHT_BRACKETS[0])}]+', DOTALL)

def normalize_punctuation_spaces(text):
    if not text:
        return text

    normalized_text = list(normalize_spaces(text))
    all_punctuations = set(ALL_PUNCTUATIONS)
    all_punctuations_with_space = set(ALL_PUNCTUATIONS + SPACE)

    for letter_index, letter in enumerate(normalized_text):
        if letter not in all_punctuations:
            continue

        if get_index(normalized_text, letter_index - 1) == SPACE:
            normalized_text[letter_index - 1] = ''

        next_letter = get_index(normalized_text, letter_index + 1, default_value=SPACE)

        if next_letter not in all_punctuations_with_space:
            normalized_text[letter_index] += SPACE

    return ''.join(normalized_text)

def replace_enters(text, repl=SPACE):
    return sub(_ENTERS_REGEX, repl, text) if text else text

def replace_protected(match, protected_letters, repl):
    value = match.group()
    protected_letters[repl].append(value)

    return repl

def replace_unprotected(match, protected_letters):
    value = match.group()

    return pop(protected_letters[value], 0)

def get_protected_text(text):
    if not text:
        return text, None

    protected_letters = {
        _PROTECTED_REPLACERS['NUMBER']: [],
        _PROTECTED_REPLACERS['MAIL']: [],
        _PROTECTED_REPLACERS['LINK']: [],
    }

    normalized_text = replace_numbers(text, lambda match: replace_protected(match, protected_letters, _PROTECTED_REPLACERS['NUMBER']))
    normalized_text = replace_dogs(normalized_text, lambda match: replace_protected(match, protected_letters, _PROTECTED_REPLACERS['MAIL']))
    normalized_text = replace_urls(normalized_text, lambda match: replace_protected(match, protected_letters, _PROTECTED_REPLACERS['LINK']))

    return normalized_text, protected_letters

def get_unprotected_text(text, protected_letters):
    normalized_text = sub(_PROTECTED_REPLACERS['LINK'], lambda match: replace_unprotected(match, protected_letters), text)
    normalized_text = sub(_PROTECTED_REPLACERS['MAIL'], lambda match: replace_unprotected(match, protected_letters), normalized_text)
    normalized_text = sub(_PROTECTED_REPLACERS['NUMBER'], lambda match: replace_unprotected(match, protected_letters), normalized_text)

    return normalized_text

def normalize_text(text):
    return normalize_punctuation_spaces(
        replace_enters(text)
    )

def standardize_text(text):
    return standardize_empty_brackets(
        replace_punctuation_duplications(
            standardize_punctuations(
                replace_all_wrong_letters(text)
            )
        )
    )

def normalize_dataset_text(text):
    return normalize_punctuation_spaces(
        replace_punctuation_duplications(text, is_editable_spaces=True)
    )

def standardize_punctuations(text):
    if not text:
        return text

    normalized_text = list(text)
    quotes = set(QUOTES[1:])
    dashes = set(DASHES[1:])
    left_brackets = set(LEFT_BRACKETS[1:])
    right_brackets = set(RIGHT_BRACKETS[1:])
    slashes = set(SLASHES[1:])
    tags = set(TAGS[1:])
    maths = set(MATHS[1:])
    currencies = set(CURRENCIES[1:])
    symbols = set(SYMBOLS[1:])
    sentence_ends = {EXCLAMATION_LETTER, ELLIPSIS_LETTER}

    for letter_index, letter in enumerate(normalized_text):
        if letter == SEMICOLON_LETTER:
            normalized_text[letter_index] = COMMA_LETTER
            continue

        if letter in sentence_ends:
            normalized_text[letter_index] = DOT_LETTER
            continue

        if letter in quotes:
            normalized_text[letter_index] = QUOTES[0]
            continue

        if letter in dashes:
            normalized_text[letter_index] = DASHES[0]
            continue

        if letter in left_brackets:
            normalized_text[letter_index] = LEFT_BRACKETS[0]
            continue

        if letter in right_brackets:
            normalized_text[letter_index] = RIGHT_BRACKETS[0]
            continue

        if letter in slashes:
            normalized_text[letter_index] = SLASHES[0]
            continue

        if letter in tags:
            normalized_text[letter_index] = TAGS[0]
            continue

        if letter in maths:
            normalized_text[letter_index] = MATHS[0]
            continue

        if letter in currencies:
            normalized_text[letter_index] = CURRENCIES[0]
            continue

        if letter in symbols:
            normalized_text[letter_index] = SYMBOLS[0]
            continue

    return ''.join(normalized_text)

def replace_punctuation_duplications(text, repl='', is_editable_spaces=False):
    if not text:
        return text

    normalized_text = normalize_spaces(text) if is_editable_spaces else text
    normalized_text = list(normalized_text)
    dashes = set(DASHES)
    base_chars = set(UNIQUE_BASE_CHARS)
    all_punctuations = set(ALL_PUNCTUATIONS)
    end_sentence_punctuations = set(END_SENTENCE)

    for letter_index, letter in enumerate(normalized_text[1:], start=1):
        if letter not in base_chars:
            continue

        prev_index = letter_index - 1
        prev_letter = get_index(normalized_text, prev_index, default_value=SPACE)

        if letter == prev_letter:
            normalized_text[prev_index] = repl
            continue

        is_letter_punctuation = letter in all_punctuations

        if is_letter_punctuation and prev_letter in all_punctuations:
            normalized_text[letter_index] = normalized_text[prev_index]
            normalized_text[prev_index] = repl
            continue

        if not is_editable_spaces or prev_letter != SPACE or not prev_index:
            continue

        prev_previous_index = letter_index - 2
        prev_previous_value = get_index(normalized_text, prev_previous_index, default_value=SPACE)

        if letter == prev_previous_value:
            normalized_text[prev_index] = repl
            normalized_text[prev_previous_index] = repl
            continue

        if is_letter_punctuation and prev_previous_value in all_punctuations:
            normalized_text[letter_index] = normalized_text[prev_previous_index]
            normalized_text[prev_index] = repl
            normalized_text[prev_previous_index] = repl
            continue

        if letter in dashes and prev_previous_value in end_sentence_punctuations:
            normalized_text[letter_index] = normalized_text[prev_previous_index]
            normalized_text[prev_index] = repl
            normalized_text[prev_previous_index] = repl
            continue

    return ''.join(normalized_text)

def replace_numbers(text, repl=DIGITS[0]):
    return sub(_NUMBERS_REGEX, repl, text) if text else text

def replace_dogs(text, repl=SPACE):
    return sub(_STANDARDIZATION_DOGS_REGEX, repl, text) if text else text

def replace_urls(text, repl=SPACE):
    return sub(_STANDARDIZATION_URLS_REGEX, repl, text) if text else text

def standardize_empty_brackets(text, repl=''):
    return sub(_STANDARDIZATION_NUMBER_WITH_BRACKETS_REGEX, repl, text) if text else text

def replace_all_unknown_letters(text, allowed_letters, repl='', with_skip_unknown_word_ending=False):
    if not text:
        return text

    need_skip_word = False
    normalized_text = list(text)
    allowed_letters = set(allowed_letters.lower())

    for letter_index, letter in enumerate(normalized_text):
        if letter == SPACE:
            need_skip_word = False

            continue

        if need_skip_word:
            normalized_text[letter_index] = repl

            continue

        if not letter.lower() in allowed_letters:
            need_skip_word = with_skip_unknown_word_ending
            normalized_text[letter_index] = repl

    return ''.join(normalized_text)

def replace_all_wrong_letters(text, repl='', with_skip_unknown_word_ending=False):
    return replace_all_unknown_letters(text, UNIQUE_ALL_LETTERS, repl, with_skip_unknown_word_ending=with_skip_unknown_word_ending)

def replace_all_punctuation_letters(text, repl='', with_skip_unknown_word_ending=False):
    return replace_all_unknown_letters(text, UNIQUE_ALPHABET_LETTERS + DIGITS, repl, with_skip_unknown_word_ending=with_skip_unknown_word_ending)

def capitalize_text(text):
    need_capitalization = True
    text_normalized = list(text)
    start_capitalizations = set(LEFT_BRACKETS + END_SENTENCE)

    for letter_index, letter in enumerate(text_normalized):
        if letter.isnumeric() or letter.isupper():
            need_capitalization = False

            continue

        if need_capitalization and letter.islower():
            need_capitalization = False
            text_normalized[letter_index] = letter.upper()

            continue

        if letter in start_capitalizations:
            need_capitalization = True

    return ''.join(text_normalized)

def get_invalid_punctuation_words(text):
    words = []

    for word in text.split():
        punctuation_indexes = [index_letter for index_letter, letter in enumerate(word) if letter in ALL_PUNCTUATIONS]
        number_indexes = len(punctuation_indexes)

        if not number_indexes:
            continue

        if punctuation_indexes[0] != len(word) - 1:
            words.append(word)

    return words

def normalize_start_text(text):
    if not text:
        return text

    forbidden_letters = set(ALL_PUNCTUATIONS + SPACE)
    text_start_pos = find_index(text, lambda letter: letter not in forbidden_letters)

    return text[text_start_pos:]

def normalize_end_text(text):
    if not text:
        return text

    last_punctuation = DOT_LETTER
    end_sentence_punctuations = set(END_SENTENCE)
    forbidden_letters = set(BASE_PUNCTUATIONS + SPACE)

    for index in range(len(text) - 1, -1, -1):
        letter = text[index]

        if letter in forbidden_letters:
            continue

        if letter in end_sentence_punctuations:
            last_punctuation = letter
            continue

        return text[:index + 1] + last_punctuation

    return text.strip()

def get_token_label(name):
    return f'</{name.lower()}/>'

def check_is_end_sentence_punctuation(letter):
    return letter == DOT_LETTER or letter == QUESTION_MARK

def check_is_base_sentence_punctuation(letter):
    return letter == COMMA_LETTER

def get_text_sentence_lengths(text):
    sentence_words = []
    sentence_lengths = dict()
    end_sentence_punctuations = set(END_SENTENCE)
    normalized_words = normalize_spaces(text).split()

    for word in normalized_words:
        last_word_letter = get_last_word_letter(word)
        sentence_words.append(word)

        if last_word_letter not in end_sentence_punctuations:
            continue

        sentence_length = len(sentence_words)
        sentence_words = []
        sentence_lengths[sentence_length] = sentence_lengths.get(sentence_length, 0) + 1

    return sentence_lengths

def normalize_text_sentences(text, min_sentence_length=None, max_sentence_length=None, exclude_enter_sentences=False):
    sentences = []
    sentence_words = []
    end_sentence_punctuations = set(END_SENTENCE)
    normalized_words = normalize_spaces(text).split(SPACE)

    for word in normalized_words:
        last_word_letter = get_last_word_letter(word)
        sentence_words.append(word)

        if last_word_letter not in end_sentence_punctuations:
            continue

        sentence_length = len(sentence_words)
        new_sentence = SPACE.join(sentence_words)
        sentence_words = []

        if min_sentence_length is not None and sentence_length < min_sentence_length:
            continue

        if max_sentence_length is not None and sentence_length > max_sentence_length:
            continue

        if exclude_enter_sentences and search(_ENTERS_REGEX, new_sentence):
            continue

        sentences.append(new_sentence)

    return SPACE.join(sentences)

def get_text_without_first_sentence(text):
    if not text:
        return text

    end_sentence_punctuations = set(END_SENTENCE)
    finish_sentence_index = find_index(list(text), lambda letter: letter in end_sentence_punctuations)
    text_start_pos = finish_sentence_index + 1 if finish_sentence_index is not None else None

    return normalize_start_text(text[text_start_pos:])

def get_text_without_last_sentence(text):
    if not text:
        return text

    end_sentence_punctuations = set(END_SENTENCE)
    finish_sentence_index = find_rindex(list(text), lambda letter: letter in end_sentence_punctuations)
    text_end_pos = finish_sentence_index + 1 if finish_sentence_index is not None else None

    return normalize_end_text(text[:text_end_pos])

def get_last_word_letter(word):
    return standardize_punctuations(word[-1]) if word else word

def get_base_punctuation_sentences(text):
    words = []
    sentence_words = []
    has_base_sentence_punctuation = False
    base_sentence_punctuations = set(BASE_PUNCTUATIONS)
    end_sentence_punctuations = set(END_SENTENCE)
    normalized_words = normalize_spaces(text).split(SPACE)

    for word in normalized_words:
        last_word_letter = get_last_word_letter(word)
        sentence_words.append(word)

        if last_word_letter in base_sentence_punctuations:
            has_base_sentence_punctuation = True

        if last_word_letter not in end_sentence_punctuations:
            continue

        if has_base_sentence_punctuation:
            words.extend(sentence_words)

        sentence_words = []
        has_base_sentence_punctuation = False

    return SPACE.join(words)

from transformers import AutoTokenizer

from utils.array import padding_array
from utils.line_text import (check_is_base_sentence_punctuation, check_is_end_sentence_punctuation,
                             standardize_punctuations)

from constants.punctuation import COMMA_PUNCTUATION_DICTIONARY, DOT_PUNCTUATION_DICTIONARY, PUNCTUATION_TYPES

def get_tokenizer(model_options, special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(model_options['BASIS'])

    if special_tokens:
        tokenizer.add_special_tokens(dict(additional_special_tokens=special_tokens))

    return tokenizer

def remove_punctuation_normalized_word(word):
    if not word:
        return word

    last_word_letter = standardize_punctuations(word[-1])
    has_end_sentence_punctuation = check_is_end_sentence_punctuation(last_word_letter)
    has_base_sentence_punctuation = check_is_base_sentence_punctuation(last_word_letter)

    if has_end_sentence_punctuation or has_base_sentence_punctuation:
        return remove_punctuation_normalized_word(word[:-1])

    return word

def remove_punctuation_text(text, punctuation_type):
    new_letters = []

    for letter in list(text):
        if check_is_base_sentence_punctuation(letter):
            continue

        if punctuation_type == PUNCTUATION_TYPES['DOT'] and check_is_end_sentence_punctuation(letter):
            continue

        new_letters.append(letter)

    return ''.join(new_letters)

def get_ended_token_sequences(model_options, x, y_mask, y=None, is_train=False):
    end_token = model_options['SEP']

    if x[-1] == end_token:
        return x, y_mask, y

    mask_value = [1] if is_train else [0]
    new_x = x + [end_token]
    new_mask = y_mask + mask_value
    new_y = y + [0] if y is not None else None

    return new_x, new_mask, new_y

def get_started_token_sequences(model_options, x, y_mask, y=None, is_train=False):
    mask_value = [1] if is_train else [0]
    new_x = [model_options['CLS']] + x
    new_mask = mask_value + y_mask
    new_y = [0] + y if y is not None else None

    return new_x, new_mask, new_y

def get_normalized_token_sequences(model_options, x, y_mask, y=None):
    input_length = model_options['INPUT_LENGTH']
    pad_option = model_options['PAD']
    x_normalized = padding_array(x, input_length, padding_value=pad_option)
    y_mask_normalized = padding_array(y_mask, input_length, padding_value=0)
    attn_mask = [1 if token != pad_option else 0 for token in x_normalized]
    y_normalized = padding_array(y, input_length, padding_value=0) if y is not None else None

    return x_normalized, attn_mask, y_mask_normalized, y_normalized

def get_punctuation_dictionary(punctuation_type):
    return DOT_PUNCTUATION_DICTIONARY if punctuation_type == PUNCTUATION_TYPES['DOT'] else COMMA_PUNCTUATION_DICTIONARY

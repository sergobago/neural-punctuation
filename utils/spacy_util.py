from utils.array import join_parameters
from utils.line_text import (check_is_base_sentence_punctuation, check_is_end_sentence_punctuation,
                             get_token_label, standardize_punctuations)

from constants.letters import ALL_PUNCTUATIONS
from constants.punctuation import PUNCTUATION_TYPES
from constants.spacy import SPACY_ENTITY_IOBS

def get_punctuation_words(model, text, punctuation_type):
    words = []

    for token in model(text):
        new_word = get_word_token(token, punctuation_type)
        words.append(new_word)

    return words

def get_punctuation_targets(model, text, punctuation_type):
    targets = []

    for word in model.tokenizer(text):
        last_word_letter = standardize_punctuations(word.text[-1])
        has_target_value = check_is_end_sentence_punctuation(last_word_letter) if punctuation_type == PUNCTUATION_TYPES['DOT'] else check_is_base_sentence_punctuation(last_word_letter)
        target_value = last_word_letter if has_target_value else ''
        targets.append(target_value)

    return targets

def get_text_token_words(model, text):
    words = []

    for token in model.tokenizer(text):
        words.append(token.text)

    return words

def get_special_tokens(model):
    entity_types = model.pipe_labels['ner']
    entity_labels = [get_entity_token(ent_type, ent_iob) for ent_type in entity_types for ent_iob in SPACY_ENTITY_IOBS]
    special_tokens = [get_token_label(name) for name in entity_labels]

    return special_tokens

def get_word_token(token, punctuation_type):
    text = token.text
    ent_token = get_entity_token(token.ent_type_, token.ent_iob_)

    if ent_token:
        ent_label = get_token_label(ent_token)
        last_word_letter = standardize_punctuations(text[-1])
        text = ent_label + last_word_letter if last_word_letter in ALL_PUNCTUATIONS else ent_label

    return join_parameters([text], separator='')

def get_entity_token(ent_type, ent_iob):
    return ent_type + ent_iob if ent_type else None

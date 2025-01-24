from copy import deepcopy

from nn.punctuation.predict import predict
from nn.punctuation.tokenizer import (check_is_base_sentence_punctuation, check_is_end_sentence_punctuation,
                                      remove_punctuation_normalized_word, remove_punctuation_text)
from utils.array import get_index
from utils.line_text import (get_protected_text, get_unprotected_text, normalize_end_text, normalize_start_text,
                             normalize_text, standardize_punctuations, standardize_text)
from utils.punctuation_neural_network import PunctuationNeuralNetwork
from utils.spacy_neural_network import SpacyNeuralNetwork
from utils.spacy_util import get_punctuation_words, get_special_tokens, get_text_token_words

from constants.letters import ALL_PUNCTUATIONS, SPACE
from constants.punctuation import PUNCTUATION_TYPES

_SPACY_INSTANCE = SpacyNeuralNetwork()
_PUNCTUATION_INSTANCE = PunctuationNeuralNetwork()

def inference(language, text, punctuation_type):
    normalized_text, protected_letters = get_protected_text(text)
    normalized_text = normalize_text(normalized_text)
    normalized_text = normalize_end_text(normalize_start_text(normalized_text))
    standardized_text = standardize_text(normalized_text)
    standardized_text = standardized_text.strip().lower()
    standardized_text = remove_punctuation_text(standardized_text, punctuation_type)
    is_base_punctuation_type = punctuation_type == PUNCTUATION_TYPES['COMMA']

    with _SPACY_INSTANCE.lock():
        spacy_model = _SPACY_INSTANCE.get_model(language)
        spacy_special_tokens = get_special_tokens(spacy_model)
        normalized_words = get_text_token_words(spacy_model, normalized_text)
        words = get_punctuation_words(spacy_model, standardized_text, punctuation_type)

    with _PUNCTUATION_INSTANCE.lock():
        punctuation_tokenizer = _PUNCTUATION_INSTANCE.get_tokenizer(language, punctuation_type, special_tokens=spacy_special_tokens)
        punctuation_model = _PUNCTUATION_INSTANCE.get_model(punctuation_tokenizer, language, punctuation_type)
        predicted_punctuations = predict(language, words, punctuation_type, preloaded_tokenizer=punctuation_tokenizer, preloaded_model=punctuation_model)

    for word_index, word in enumerate(normalized_words):
        last_word_letter = standardize_punctuations(word[-1])
        has_base_sentence_punctuation = check_is_base_sentence_punctuation(last_word_letter)
        predicted_punctuation = get_index(predicted_punctuations, word_index)

        if not predicted_punctuation and has_base_sentence_punctuation and is_base_punctuation_type:
            normalized_words[word_index] = remove_punctuation_normalized_word(normalized_words[word_index])

        if predicted_punctuation and last_word_letter not in ALL_PUNCTUATIONS:
            normalized_words[word_index] = remove_punctuation_normalized_word(normalized_words[word_index]) + predicted_punctuation

    normalized_words[-1] = normalize_end_text(normalized_words[-1])
    normalized_text = get_unprotected_text(normalized_text, deepcopy(protected_letters))
    punctuated_text = SPACE.join(normalized_words)
    punctuated_text = get_unprotected_text(punctuated_text, protected_letters)

    return punctuated_text, normalized_text

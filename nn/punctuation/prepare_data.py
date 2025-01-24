from datetime import datetime
from functools import partial
from json import dumps, loads
from multiprocessing import Pool

from nn.punctuation.tokenizer import (check_is_end_sentence_punctuation, get_ended_token_sequences,
                                      get_punctuation_dictionary, get_started_token_sequences, remove_punctuation_text)
from utils.array import concatenate_arrays, get_array_parts
from utils.line_text import get_text_without_first_sentence, get_text_without_last_sentence
from utils.spacy_neural_network import SpacyNeuralNetwork
from utils.spacy_util import get_punctuation_targets, get_punctuation_words

from constants.project_params import ENCODING
from constants.punctuation import PUNCTUATION_TYPES
from constants.spacy import MAX_SPACY_CHUNK_SIZE, SPACY_CPU_CORES

def save_dataset_words(language, punctuation_type, input_path, output_path):
    batch_size = MAX_SPACY_CHUNK_SIZE * SPACY_CPU_CORES

    with open(input_path, 'r', encoding=ENCODING) as readable_file:
        for chunk in iter(lambda: readable_file.read(batch_size), ''):
            text_parts = get_array_parts(chunk, SPACY_CPU_CORES)

            with Pool(SPACY_CPU_CORES) as processes:
                lines = processes.map(partial(get_words_lines, language=language, punctuation_type=punctuation_type), text_parts)
                lines = concatenate_arrays(lines)

            print('write_words', len(lines), datetime.now())

            with open(output_path, 'a', encoding=ENCODING) as writable_file:
                for line in lines:
                    writable_file.write(line)

def get_words_lines(text, language, punctuation_type):
    spacy_model = SpacyNeuralNetwork().load_model(language)
    print('_get_words_lines', len(text), '/', MAX_SPACY_CHUNK_SIZE, datetime.now())
    normalized_text = get_text_without_first_sentence(text)
    normalized_text = get_text_without_last_sentence(normalized_text)
    normalized_text = normalized_text.strip().lower()
    targets = get_punctuation_targets(spacy_model, normalized_text, punctuation_type)
    normalized_text = remove_punctuation_text(normalized_text, punctuation_type)
    words = get_punctuation_words(spacy_model, normalized_text, punctuation_type)

    assert len(words) == len(targets), f'Длины массивов линий не соответствуют {len(words)}, {len(targets)}'

    return [f'{word}\t{target}\n' for word, target in zip(words, targets)]

def get_data_items(model_options, tokenizer, words, punctuation_type, targets=None):
    x = []
    y = []
    y_mask = []
    data_items = []
    has_targets = targets is not None
    option_unk = model_options['UNK']
    input_length = model_options['INPUT_LENGTH']
    punctuation_dictionary = get_punctuation_dictionary(punctuation_type)
    max_allowed_input_length = input_length - 1
    end_word_position = max_allowed_input_length
    end_sentence_position = max_allowed_input_length
    is_comma_punctuation = punctuation_type == PUNCTUATION_TYPES['COMMA']
    x, y_mask, y = get_started_token_sequences(model_options, x, y_mask, y=y)

    for word_index, word in enumerate(words):
        last_word_letter = word[-1]
        tokens = tokenizer.tokenize(word)
        length_tokens = len(tokens)
        target = punctuation_dictionary[targets[word_index]] if has_targets else 0

        if len(x) + length_tokens > max_allowed_input_length:
            end_position = min(max_allowed_input_length, end_sentence_position, end_word_position)
            new_sequence = get_ended_token_sequences(model_options, x[:end_position], y_mask[:end_position], y=y[:end_position])
            data_items.append(new_sequence)
            x, y_mask, y = get_started_token_sequences(model_options, x[end_position:], y_mask[end_position:], y=y[end_position:])
            end_sentence_position = max_allowed_input_length
            assert len(new_sequence[0]) <= input_length, f'Переполнена последовательность входных данных {len(new_sequence[0])}'

        for index_token in range(length_tokens - 1):
            x.append(tokenizer.convert_tokens_to_ids(tokens[index_token]))
            y.append(0)
            y_mask.append(0)

        x.append(tokenizer.convert_tokens_to_ids(tokens[-1]) if length_tokens else option_unk)
        y.append(target)
        y_mask.append(1)

        if is_comma_punctuation and check_is_end_sentence_punctuation(last_word_letter):
            x, y_mask, y = get_ended_token_sequences(model_options, x, y_mask, y=y)
            end_sentence_position = len(x)

        end_word_position = len(x)

    if len(x) > 1 and not has_targets:
        new_sequence = get_ended_token_sequences(model_options, x, y_mask, y=y)
        data_items.append(new_sequence)

    return data_items

def save_data_items(output_path, data_items):
    with open(output_path, 'a', encoding=ENCODING) as writable_file:
        for data_item in data_items:
            line = dumps(data_item) + '\n'
            writable_file.write(line)

def get_next_data_item(data):
    next_data_item = next(data).strip()

    return loads(next_data_item)

from csv import DictReader, field_size_limit
from multiprocessing import Pool
from os import listdir
from os.path import dirname, join

from utils.array import find_rindex, get_array_step_parts, safeget
from utils.helpers import find, some
from utils.json_parse import json_loads
from utils.line_text import (get_base_punctuation_sentences, get_invalid_punctuation_words, get_protected_text,
                             get_text_sentence_lengths, get_text_without_first_sentence, normalize_dataset_text,
                             normalize_end_text, normalize_start_text, normalize_text, normalize_text_sentences,
                             standardize_text)
from utils.path import get_path_extension, get_path_filename
from utils.strip_tags import normalize_spaces, strip_special_characters, strip_tags

from constants.letters import DOT_LETTER, SPACE
from constants.ml_punctuation import (DATASET_DATA_KEYS, DATASET_DEFAULT_EXTENSION, DATASET_NORMALIZED_EXTENSION,
                                      DATASET_PARSE_EXTENSIONS, NUMBER_USED_CPU_CORES)
from constants.project_params import ENCODING
from constants.punctuation import PRETRAINED_MODELS

_MAX_SENTENCE_LENGTH = 45
_CHUNK_FILE_SIZE = int(1e8)
_WRITING_FILE_SIZE = int(1e6)
field_size_limit(_CHUNK_FILE_SIZE)

def parse_json_file(path):
    with open(path, 'r', encoding=ENCODING) as json_file:
        texts = [get_parsed_text(json_loads(json_str)) for json_str in list(json_file)]

    write_file_texts(path, texts)

def parse_csv_file(path):
    with open(path, 'r', encoding=ENCODING) as csv_file:
        texts = [get_parsed_text(csv_item) for csv_item in DictReader(csv_file)]

    write_file_texts(path, texts)

def get_parsed_text(data_item):
    keys = find(DATASET_DATA_KEYS, lambda keys: safeget(data_item, keys))
    text = safeget(data_item, keys, '')

    return normalize_end_text(normalize_start_text(text))

def write_file_texts(path, texts, extension='txt'):
    filename = get_path_filename(path, extension)
    output_path = join(dirname(path), filename)
    chunks = get_array_step_parts(texts, _WRITING_FILE_SIZE)

    for chunk_index, chunk in enumerate(chunks):
        mode = 'a' if chunk_index else 'w'
        text = SPACE.join(chunk) + SPACE

        with open(output_path, mode, encoding=ENCODING) as output_file:
            output_file.write(text)

    print('Written', output_path)

def normalize_text_file(path):
    output_filename = get_path_filename(path, DATASET_NORMALIZED_EXTENSION)
    output_path = join(dirname(path), output_filename)

    with open(path, 'r', encoding=ENCODING) as input_file:
        for chunk_index, chunk in enumerate(iter(lambda: input_file.read(_CHUNK_FILE_SIZE), '')):
            mode = 'a' if chunk_index else 'w'
            normalized_text = strip_special_characters(strip_tags(chunk))
            normalized_text = get_text_without_first_sentence(normalized_text)
            normalized_text = get_protected_text(normalized_text)[0]
            normalized_text = normalize_text_sentences(normalized_text, exclude_enter_sentences=True)
            normalized_text = normalize_text(normalized_text)
            normalized_text = standardize_text(normalized_text)
            normalized_text = normalize_dataset_text(normalized_text)
            normalized_text = normalize_text_sentences(normalized_text, max_sentence_length=_MAX_SENTENCE_LENGTH)
            normalized_text = get_base_punctuation_sentences(normalized_text)
            text_sentence_lengths = get_text_sentence_lengths(normalized_text)
            invalid_punctuation_words = get_invalid_punctuation_words(normalized_text)
            max_text_sentence_length = max(text_sentence_lengths.keys())

            print('Map lengths', text_sentence_lengths)
            print(f'Normalized {path} step {chunk_index + 1}')

            assert max_text_sentence_length <= _MAX_SENTENCE_LENGTH, f'Слишком много слов в предложении: {max_text_sentence_length}'
            assert len(invalid_punctuation_words) == 0, f'Ошибки пунктуации в словах: {invalid_punctuation_words}'

            with open(output_path, mode, encoding=ENCODING) as output_file:
                output_file.write(normalized_text)

    strip_text_file(output_path)

def strip_text_file(path):
    with open(path, 'r', encoding=ENCODING) as file:
        normalized_text = normalize_start_text(normalize_spaces(file.read()))
        text_end_pos = find_rindex(normalized_text, DOT_LETTER, default_value=0)
        normalized_text = normalized_text[:text_end_pos + 1]

    with open(path, 'w', encoding=ENCODING) as file:
        file.write(normalized_text)

    print('Striped', path)

def merge_text_files(file_paths, output_path):
    for path_index, path in enumerate(file_paths):
        mode = 'a' if path_index else 'w'

        with open(path, 'r', encoding=ENCODING) as input_file:
            text = normalize_end_text(normalize_start_text(input_file.read()))

        with open(output_path, mode, encoding=ENCODING) as output_file:
            output_file.write(text + SPACE)

    print('Merged text files')

def parse_datasets(dataset_dir):
    dataset_paths = [join(dataset_dir, file) for file in listdir(dataset_dir) if some(DATASET_PARSE_EXTENSIONS, lambda extension: file.endswith(DOT_LETTER + extension))]

    with Pool(NUMBER_USED_CPU_CORES) as processes:
        processes.map(parse_dataset, dataset_paths)

def normalize_datasets(dataset_dir):
    default_extension = DOT_LETTER + DATASET_DEFAULT_EXTENSION
    filenames = listdir(dataset_dir)
    dataset_paths = [join(dataset_dir, name) for name in filenames if name.endswith(default_extension)]

    with Pool(NUMBER_USED_CPU_CORES) as processes:
        processes.map(normalize_text_file, dataset_paths)

def merge_train_datasets(dataset_dir, language):
    output_path = join(dataset_dir, PRETRAINED_MODELS[language]['DATASET_TRAIN'])
    merge_datasets(dataset_dir, output_path, DATASET_NORMALIZED_EXTENSION)

def merge_datasets(dataset_dir, output_path, extension):
    normalized_extension = DOT_LETTER + extension
    filenames = listdir(dataset_dir)
    dataset_paths = [join(dataset_dir, name) for name in filenames if name.endswith(normalized_extension)]
    merge_text_files(dataset_paths, output_path)

def parse_dataset(dataset_path):
    parser = dict(json=parse_json_file, jsonl=parse_json_file, csv=parse_csv_file)
    extension = get_path_extension(dataset_path)
    dataset_parser = parser.get(extension)
    dataset_parser(dataset_path)

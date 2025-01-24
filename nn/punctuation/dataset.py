from datetime import datetime
from functools import partial
from multiprocessing import Pool
from os.path import join
from spacy import require_cpu, require_gpu
from torch import tensor, utils

from nn.punctuation.augmentation import get_augment
from nn.punctuation.prepare_data import get_data_items, get_next_data_item, save_data_items, save_dataset_words
from nn.punctuation.tokenizer import get_normalized_token_sequences, get_punctuation_dictionary
from utils.array import concatenate_arrays, find_index, get_array_parts
from utils.file import get_file_number_lines, is_file_exist

from constants.letters import DOT_LETTER
from constants.ml_punctuation import AUGMENT_RATE_COMMA, AUGMENT_RATE_DOT, NUMBER_USED_CPU_CORES
from constants.project_params import DATASET_DIR, ENCODING, TEMP_DIR
from constants.punctuation import PRETRAINED_MODELS, PUNCTUATION_TYPES

_DATASET_CHUNK_SIZE = int(1e9)

class Dataset(utils.data.Dataset):
    def __init__(self, language, tokenizer, punctuation_type, is_train=False):
        self.language = language
        self.punctuation_type = punctuation_type
        self.punctuation_dictionary = get_punctuation_dictionary(self.punctuation_type)
        self.model_options = PRETRAINED_MODELS[self.language]
        self.option_pad = self.model_options['PAD']
        self.option_unk = self.model_options['UNK']
        self.option_cls = self.model_options['CLS']
        self.option_sep = self.model_options['SEP']
        self.input_length = self.model_options['INPUT_LENGTH']
        self.dataset_filename = self.model_options['DATASET_TRAIN'] if is_train else self.model_options['DATASET_VALIDATION']
        self.dataset_text_path = join(DATASET_DIR, self.language, self.dataset_filename)
        self.dataset_word_path = join(TEMP_DIR, self.language, self.punctuation_type, f'word_{self.dataset_filename}')
        self.dataset_data_path = join(TEMP_DIR, self.language, self.punctuation_type, f'data_{self.dataset_filename}')
        self.augment_rate = AUGMENT_RATE_DOT if punctuation_type == PUNCTUATION_TYPES['DOT'] else AUGMENT_RATE_COMMA
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.data = self._load_data()
        self.data_length = get_file_number_lines(self.dataset_data_path, encoding=ENCODING)  # len(self.data)

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        x, y_mask, y = get_next_data_item(self.data)  # self.data[index]

        if self.is_train and self.augment_rate:
            x, y_mask, y = get_augment(self.model_options, self.tokenizer, self.augment_rate, x, y_mask, y)

        x, attn_mask, y_mask, y = get_normalized_token_sequences(self.model_options, x, y_mask, y)

        return tensor(x), tensor(attn_mask), tensor(y_mask), tensor(y)

    def _load_data(self):
        require_gpu()

        if not is_file_exist(self.dataset_word_path):
            save_dataset_words(self.language, self.punctuation_type, self.dataset_text_path, self.dataset_word_path)

        require_cpu()

        if not is_file_exist(self.dataset_data_path):
            with open(self.dataset_word_path, 'r', encoding=ENCODING) as readable_file:
                for chunk in iter(lambda: readable_file.read(_DATASET_CHUNK_SIZE), ''):
                    print('_get_parsed_data_items', len(chunk), '/', _DATASET_CHUNK_SIZE, datetime.now())

                    new_data_items = self._get_parsed_data_items(chunk)

                    print('write_data', len(new_data_items), datetime.now())

                    save_data_items(self.dataset_data_path, new_data_items)

        return open(self.dataset_data_path, 'r', encoding=ENCODING)

    def _get_parsed_data_items(self, text):
        text_parts = get_array_parts(text, NUMBER_USED_CPU_CORES)

        with Pool(NUMBER_USED_CPU_CORES) as processes:
            result = processes.map(partial(self._parse_data, model_options=self.model_options, tokenizer=self.tokenizer, punctuation_type=self.punctuation_type), text_parts)

            return concatenate_arrays(result)

    @staticmethod
    def _parse_data(text, model_options, tokenizer, punctuation_type):
        words = []
        targets = []
        text_start_pos = find_index(text, DOT_LETTER)
        text_start_pos = text_start_pos + 1 if text_start_pos is not None else None
        normalized_text = text[text_start_pos:]
        normalized_text = normalized_text.strip().lower()
        lines = normalized_text.split('\n')

        for line in lines[:-1]:
            word, target = line.split('\t')
            words.append(word)
            targets.append(target)

        return get_data_items(model_options, tokenizer, words, punctuation_type, targets=targets)

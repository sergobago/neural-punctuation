# -*- coding: utf8 -*-
from os.path import join

from nn.punctuation.inference import inference
from nn.punctuation.train import fit
from utils.ml_dataset import parse_datasets, normalize_datasets, merge_train_datasets
from constants.languages import RU
from constants.project_params import DATASET_DIR
from constants.punctuation import PUNCTUATION_TYPES

if __name__ == '__main__':
    language = RU
    dataset_dir = join(DATASET_DIR, language)
    print('dataset_dir', dataset_dir)

    # Подготовка датасетов
    parse_datasets(dataset_dir) # Распарсить и сохранить датасеты в файлы *.txt
    normalize_datasets(dataset_dir) # Обработать текстовые датасеты в файлы *.normalized
    merge_train_datasets(dataset_dir, language) # Объединить обработанные датасеты в файл summary_train.dataset

    # Тренировка нейронной сети: COMMA - запятые, DOT - точки/знаки вопроса
    fit(language, PUNCTUATION_TYPES['COMMA']) # Чем больше гигабайт обработанный датасет, тем лучше результат

    # Тестирование натренированной модели
    predict = inference(language, 'Привет как дела учитель', PUNCTUATION_TYPES['COMMA'])
    print('Результат: ' + predict[0])

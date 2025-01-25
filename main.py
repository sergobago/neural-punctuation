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

    # Обработка датасетов
    # parse_datasets(dataset_dir) # Преобразовать датасеты *.csv, *.json в файлы *.txt
    # normalize_datasets(dataset_dir) # Обработать тексты датасетов  *.txt в текстовые файлы *.normalized
    # merge_train_datasets(dataset_dir, language) # Объединить датасеты *.normalized в файл summary_train.dataset

    # Обучение нейронной сети для заданного языка и выбранного типа пунктуации, где COMMA - запятые, DOT - точки/знаки вопроса
    # При обучении на слишком маленьких датасетах у модели будет val_acc=0.00000000
    # На датасетах любого языка от нескольких гигабайт нужно максимум 1 или 2 эпохи, дальше переобучение и ухудшение результата
    fit(language, PUNCTUATION_TYPES['COMMA']) # Чем больше гигабайт ОБРАБОТАННЫЙ датасет, тем лучше результат расстановки запятых, иначе результат будет плохой

    # Расстанавка запятых с помощью обученной модели в папке nmodels/COMMA/язык
    # predict = inference(language, 'Привет как дела друг', PUNCTUATION_TYPES['COMMA'])
    # print('Результат: ' + predict[0])

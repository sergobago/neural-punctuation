# Punctuation restoration using Transformer model
Project to train fast and accurate punctuation restoration neural network model for production on website.

## Prerequirements
* Python 3.9 for training

## Installation
```
python -m venv venv;
venv\Scripts\activate;
pip install pipenv;
pipenv install --dev;
```

## Demo
https://tool-tube.com/punctuation

## Main.py
#### Dataset processing
```
    parse_datasets(dataset_dir) # Преобразовать датасеты *.csv, *.json в файлы *.txt
    normalize_datasets(dataset_dir) # Обработать тексты датасетов  *.txt в текстовые файлы *.normalized
    merge_train_datasets(dataset_dir, language) # Объединить датасеты *.normalized в файл summary_train.dataset
```
#### Training
```
    fit(language, PUNCTUATION_TYPES['COMMA']) # Чем больше гигабайт ОБРАБОТАННЫЙ датасет, тем лучше результат расстановка запятых, иначе результат будет плохой
```
#### Inference
```
    predict = inference(language, 'Привет как дела друг', PUNCTUATION_TYPES['COMMA'])
    print('Результат: ' + predict[0])
```

## Model architecture
1) Pre-trained model FacebookAI/xlm-roberta-base
2) Bi-LSTM layer
3) Linear layer

## Credits
Our article on [habr.ru](https://habr.ru)

Project is written based on code from repositories: 
- [sviperm/neuro-comma](https://github.com/sviperm/neuro-comma)
- [xashru/punctuation-restoration](https://github.com/xashru/punctuation-restoration)
- [vlomme/Bert-Russian-punctuation](https://github.com/vlomme/Bert-Russian-punctuation)

У каждого языка есть папка COMMA для обучения модели расстановки запятых
У каждого языка есть папка DOT для обучения модели расстановки точек/знаков вопросов

Файл word_summary_train.dataset содержит весь тренировочный датасет по слову в строке, каждое слово было обработано библиотекой Spacy
Файл word_summary_validation.dataset содержит весь валидационный датасет по слову в строке, каждое слово было обработано библиотекой Spacy
Файл data_summary_train.dataset содержит данные из файла word_summary_train.dataset, которые сразу будут использоваться для обучения нейронной сети
Файл data_summary_validation.dataset содержит данные из файла word_summary_validation.dataset, которые сразу будут использоваться для валидации нейронной сети

УДАЛИТЕ СТАРЫЕ ФАЙЛЫ word_summary_train.dataset, word_summary_validation.dataset, data_summary_train.dataset, data_summary_validation.dataset ПЕРЕД ЗАПУСКОМ ОБУЧЕНИЯ, ЕСЛИ ВНЕСЛИ КАКИЕ-ЛИБО ИЗМЕНЕНИЯ В ДАТАСЕТЫ ИЛИ КОД ОБУЧЕНИЯ!!!

В этой папке все файлы создаются автоматически при начале обучения сети строкой fit(language, PUNCTUATION_TYPES['COMMA']) в файле main.py

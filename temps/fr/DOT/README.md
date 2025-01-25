В этой папке будут автоматически создаваться файлы перед обучением нейронной сети

word_summary_train.dataset содержит датасет /datasets/fr/summary_train.dataset, но каждое слово было обработано библиотекой Spacy и записано на отдельную строку
word_summary_validation.dataset содержит датасет /datasets/fr/summary_validation.dataset, но каждое слово было обработано библиотекой Spacy и записано на отдельную строку
data_summary_train.dataset содержит датасет word_summary_train.dataset, преобразованный в числа для обучения нейронной сети
data_summary_validation.dataset содержит датасет word_summary_validation.dataset, преобразованный в числа для обучения нейронной сети

ОЧИЩАЙТЕ ПАПКУ ПЕРЕД НАЧАЛОМ ОБУЧЕНИЯ, ЕСЛИ ВНЕСЛИ КАКИЕ-ЛИБО ИЗМЕНЕНИЯ В ДАТАСЕТЫ ИЛИ ВХОДНЫЕ ДАННЫЕ!

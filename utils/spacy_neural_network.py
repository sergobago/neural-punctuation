from os.path import join
from spacy import load
from spacy.tokens import Doc
from threading import Lock

from constants.languages import LANGUAGES
from constants.letters import SPACE
from constants.spacy import MAX_SPACY_CHUNK_SIZE, SPACY_DIR

class SpacyNeuralNetwork:
    __shared_state = dict(_models=dict(), _lock=Lock(), _resetable=True)

    def __init__(self):
        self.__dict__ = self.__shared_state

    def lock(self):
        return self._lock

    def reset(self):
        self._models = dict()

    def load_model(self, language):
        model_lang = LANGUAGES[language]
        print(f'SpacyNeuralNetwork load {model_lang}')
        model_path = join(SPACY_DIR, model_lang)
        model = load(model_path)
        model.tokenizer = WhitespaceTokenizer(model.vocab)
        model.max_length = MAX_SPACY_CHUNK_SIZE

        return model

    def get_model(self, language):
        model_lang = LANGUAGES[language]
        model = self._models.get(model_lang)

        if self._resetable and not model:
            self.reset()

        if not model:
            model = self.load_model(language)
            self._models[model_lang] = model

        print(f'SpacyNeuralNetwork use {model_lang}')

        return model

class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.strip().split(SPACE)
        spaces = [True] * len(words)

        for index, word in enumerate(words):
            if word == '':
                words[index] = SPACE
                spaces[index] = False

        return Doc(self.vocab, words=words, spaces=spaces)

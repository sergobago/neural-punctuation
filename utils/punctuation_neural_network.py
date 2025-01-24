from os import environ
from os.path import join
from threading import Lock
from torch import load

from nn.punctuation.model import DeepPunctuation
from nn.punctuation.tokenizer import get_tokenizer
from utils.ml_pytorch import get_device
from utils.ml_utils import get_last_checkpoint

from constants.languages import LANGUAGES
from constants.project_params import NMODELS_DIR
from constants.punctuation import PRETRAINED_MODELS, PUNCTUATION_TYPES

environ['TOKENIZERS_PARALLELISM'] = 'false'

class PunctuationNeuralNetwork:
    __shared_state = dict(
        _models={ PUNCTUATION_TYPES['DOT']: dict(), PUNCTUATION_TYPES['COMMA']: dict() },
        _tokenizers={ PUNCTUATION_TYPES['DOT']: dict(), PUNCTUATION_TYPES['COMMA']: dict() },
        _lock=Lock(),
        _resetable=True,
    )

    def __init__(self):
        self.__dict__ = self.__shared_state

    def lock(self):
        return self._lock

    def get_model(self, tokenizer, language, punctuation_type):
        model_lang = LANGUAGES[language]
        model = self._models[punctuation_type].get(model_lang)

        if self._resetable and not model:
            self._models[punctuation_type] = dict()

        if not model:
            print(f'PunctuationNeuralNetwork load {model_lang} {punctuation_type}')
            device = get_device()
            checkpoint_dir = join(NMODELS_DIR, punctuation_type, model_lang)
            checkpoint = get_last_checkpoint(checkpoint_dir)
            model = DeepPunctuation(tokenizer, model_lang, punctuation_type)
            model.load_state_dict(load(join(checkpoint_dir, checkpoint), map_location=device))
            self._models[punctuation_type][model_lang] = model

        print(f'PunctuationNeuralNetwork use {model_lang} {punctuation_type}')

        return model

    def get_tokenizer(self, language, punctuation_type, special_tokens=None):
        model_lang = LANGUAGES[language]
        tokenizer = self._tokenizers[punctuation_type].get(model_lang)

        if self._resetable and not tokenizer:
            self._tokenizers[punctuation_type] = dict()

        if not tokenizer:
            model_options = PRETRAINED_MODELS[model_lang]
            tokenizer = get_tokenizer(model_options, special_tokens)
            self._tokenizers[punctuation_type][model_lang] = tokenizer

        return tokenizer

from constants.languages import DE, EN, ES, FR, KZ, RU, UA

DOT_PUNCTUATION_DICTIONARY = { '': 0, '.': 1, '?': 2 }
COMMA_PUNCTUATION_DICTIONARY = { '': 0, ',': 1 }
FREEZED_PRETRAINED_MODEL = False
HIDDEN_DROPOUT = 0.1
ATTENTION_DROPOUT = 0.1

PUNCTUATION_TYPES = {
    'COMMA': 'COMMA',
    'DOT': 'DOT',
    'CAPITAL': 'CAPITAL',
}

PRETRAINED_MODELS = {
    RU: {
        'BASIS': 'FacebookAI/xlm-roberta-base',
        'DATASET_TRAIN': 'summary_train.dataset',
        'DATASET_VALIDATION': 'summary_validation.dataset',
        'INPUT_LENGTH': 256,
        'HIDDEN_DIM': 768,
        'PAD': 1,
        'UNK': 3,
        'CLS': 0,
        'SEP': 2,
    },
    EN: {
        'BASIS': 'FacebookAI/xlm-roberta-base',
        'DATASET_TRAIN': 'summary_train.dataset',
        'DATASET_VALIDATION': 'summary_validation.dataset',
        'INPUT_LENGTH': 256,
        'HIDDEN_DIM': 768,
        'PAD': 1,
        'UNK': 3,
        'CLS': 0,
        'SEP': 2,
    },
    FR: {
        'BASIS': 'FacebookAI/xlm-roberta-base',
        'DATASET_TRAIN': 'summary_train.dataset',
        'DATASET_VALIDATION': 'summary_validation.dataset',
        'INPUT_LENGTH': 256,
        'HIDDEN_DIM': 768,
        'PAD': 1,
        'UNK': 3,
        'CLS': 0,
        'SEP': 2,
    },
    DE: {
        'BASIS': 'FacebookAI/xlm-roberta-base',
        'DATASET_TRAIN': 'summary_train.dataset',
        'DATASET_VALIDATION': 'summary_validation.dataset',
        'INPUT_LENGTH': 256,
        'HIDDEN_DIM': 768,
        'PAD': 1,
        'UNK': 3,
        'CLS': 0,
        'SEP': 2,
    },
    KZ: {
        'BASIS': 'FacebookAI/xlm-roberta-base',
        'DATASET_TRAIN': 'summary_train.dataset',
        'DATASET_VALIDATION': 'summary_validation.dataset',
        'INPUT_LENGTH': 256,
        'HIDDEN_DIM': 768,
        'PAD': 1,
        'UNK': 3,
        'CLS': 0,
        'SEP': 2,
    },
    UA: {
        'BASIS': 'FacebookAI/xlm-roberta-base',
        'DATASET_TRAIN': 'summary_train.dataset',
        'DATASET_VALIDATION': 'summary_validation.dataset',
        'INPUT_LENGTH': 256,
        'HIDDEN_DIM': 768,
        'PAD': 1,
        'UNK': 3,
        'CLS': 0,
        'SEP': 2,
    },
    ES: {
        'BASIS': 'FacebookAI/xlm-roberta-base',
        'DATASET_TRAIN': 'summary_train.dataset',
        'DATASET_VALIDATION': 'summary_validation.dataset',
        'INPUT_LENGTH': 256,
        'HIDDEN_DIM': 768,
        'PAD': 1,
        'UNK': 3,
        'CLS': 0,
        'SEP': 2,
    },
}

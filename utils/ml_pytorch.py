import random
from numpy.random import seed
from torch import backends, cuda, manual_seed

def get_device():
    return 'cuda' if cuda.is_available() else 'cpu'

def set_seed(value): # Для воспроизводимости одних и тех же результатов при каждом запуске обучения
    seed(value)
    random.seed(value)
    manual_seed(value)
    cuda.manual_seed(value)
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False

def get_nlp_tokens(tokenizer):
    return {
        tokenizer.pad_token: tokenizer.pad_token_id,
        tokenizer.unk_token: tokenizer.unk_token_id,
        tokenizer.cls_token: tokenizer.cls_token_id,
        tokenizer.sep_token: tokenizer.sep_token_id,
        tokenizer.mask_token: tokenizer.mask_token_id,
    }

def get_nlp_all_tokens(tokenizer):
    return [{ special_token: tokenizer.all_special_ids[special_token_index] } for special_token_index, special_token in enumerate(tokenizer.all_special_tokens)]

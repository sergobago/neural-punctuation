from os.path import join
from torch import argmax, flatten, load, no_grad, tensor

from nn.punctuation.model import DeepPunctuation
from nn.punctuation.prepare_data import get_data_items
from nn.punctuation.tokenizer import get_normalized_token_sequences, get_punctuation_dictionary, get_tokenizer
from utils.array import invert_dictionary
from utils.ml_pytorch import get_device, get_nlp_tokens
from utils.ml_utils import get_last_checkpoint

from constants.project_params import NMODELS_DIR
from constants.punctuation import PRETRAINED_MODELS

def predict(language, words, punctuation_type, special_tokens=None, preloaded_tokenizer=None, preloaded_model=None):
    device = get_device()
    punctuation_dictionary = get_punctuation_dictionary(punctuation_type)
    punctuation_map = invert_dictionary(punctuation_dictionary)

    model = preloaded_model
    tokenizer = preloaded_tokenizer
    model_options = PRETRAINED_MODELS[language]
    checkpoint_dir = join(NMODELS_DIR, punctuation_type, language)
    checkpoint = None if model else get_last_checkpoint(checkpoint_dir)

    if not checkpoint and not model:
        return None

    if not tokenizer:
        tokenizer = get_tokenizer(model_options, special_tokens)

    if not model:
        model = DeepPunctuation(tokenizer, language, punctuation_type)
        model.load_state_dict(load(join(checkpoint_dir, checkpoint), map_location=device))

    model.to(device)
    model.eval()

    data_items = get_data_items(model_options, tokenizer, words, punctuation_type)
    predicts = get_predicts(model, tokenizer, model_options, data_items, punctuation_map, device=device)

    return predicts

def get_predicts(model, tokenizer, model_options, data_items, punctuation_map, device=None):
    predicts = []
    common_x = []
    common_attn_mask = []
    common_y_mask = []

    for x, y_mask, y in data_items:
        normalized_x, attn_mask, normalized_y_mask, y = get_normalized_token_sequences(model_options, x, y_mask)
        common_x.append(normalized_x)
        common_attn_mask.append(attn_mask)
        common_y_mask.extend(normalized_y_mask)

    common_x = tensor(common_x, device=device)
    common_attn_mask = tensor(common_attn_mask, device=device)
    common_y_mask = tensor(common_y_mask, device=device)

    with no_grad():
        y_predict = model(common_x, common_attn_mask)
        y_predict = y_predict.view(-1, y_predict.shape[2])
        y_predict = argmax(y_predict, dim=1).view(-1)

    for attention_index, attention_value in enumerate(flatten(common_attn_mask)):
        if attention_value and common_y_mask[attention_index]:
            punctuation_number = y_predict[attention_index].item()
            predicts.append(punctuation_map[punctuation_number])

    return predicts

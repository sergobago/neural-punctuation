from torch import nn, transpose
from transformers import AutoModel

from nn.punctuation.tokenizer import get_punctuation_dictionary

from constants.punctuation import ATTENTION_DROPOUT, FREEZED_PRETRAINED_MODEL, HIDDEN_DROPOUT, PRETRAINED_MODELS

class DeepPunctuation(nn.Module):
    def __init__(self, tokenizer, language, punctuation_type):
        super(DeepPunctuation, self).__init__()
        self.punctuation_type = punctuation_type
        self.model_options = PRETRAINED_MODELS[language]
        self.punctuation_dictionary = get_punctuation_dictionary(self.punctuation_type)
        self.pretrained_transformer = AutoModel.from_pretrained(self.model_options['BASIS'], hidden_dropout_prob=HIDDEN_DROPOUT, attention_probs_dropout_prob=ATTENTION_DROPOUT)
        self.pretrained_transformer.resize_token_embeddings(len(tokenizer))
        bert_dim = self.model_options['HIDDEN_DIM']

        if FREEZED_PRETRAINED_MODEL:
            for parameter in self.pretrained_transformer.parameters():
                parameter.requires_grad = False

        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=bert_dim, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=bert_dim * 2, out_features=len(self.punctuation_dictionary))

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample

        # (B, N, E) -> (B, N, E)
        x = self.pretrained_transformer(x, attention_mask=attn_masks)[0]
        # (B, N, E) -> (N, B, E)
        x = transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = transpose(x, 0, 1)
        x = self.linear(x)

        return x

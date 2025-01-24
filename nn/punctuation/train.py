import torch
from os.path import join
from tqdm import tqdm

from nn.punctuation.dataset import Dataset
from nn.punctuation.model import DeepPunctuation
from nn.punctuation.tokenizer import get_punctuation_dictionary, get_tokenizer
from utils.ml_pytorch import get_device, get_nlp_all_tokens, set_seed
from utils.ml_utils import get_checkpoint_epoch, get_last_checkpoint
from utils.spacy_neural_network import SpacyNeuralNetwork
from utils.spacy_util import get_special_tokens

from constants.ml_punctuation import BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED, WEIGHT_DECAY
from constants.project_params import NMODELS_DIR
from constants.punctuation import PRETRAINED_MODELS

torch.set_printoptions(profile='full')

def validate(device, model, loss_fn, optimizer, data_loader):
    model.eval()
    sum_correct = 0
    sum_total = 0
    sum_loss = 0
    number_iterations = 0

    with torch.no_grad():
        for x, attn_mask, y_mask, y in tqdm(data_loader, desc='validate'):
            x = x.to(device)
            y = y.to(device)
            attn_mask = attn_mask.to(device)
            y_mask = y_mask.to(device)
            y = y.view(-1)
            y_mask = y_mask.view(-1)

            y_predict = model(x, attn_mask)
            y_predict = y_predict.view(-1, y_predict.shape[2])
            loss = loss_fn(y_predict, y)
            y_predict = torch.argmax(y_predict, dim=1).view(-1)
            sum_correct += torch.sum(y_mask * ((y > 0) + (y_predict > 0)) * (y_predict == y)).item()
            sum_total += torch.sum(y_mask * ((y > 0) + (y_predict > 0))).item()
            sum_loss += loss.item()
            number_iterations += 1

    return sum_loss/number_iterations, sum_correct/sum_total

def train(device, model, loss_fn, optimizer, data_loader):
    model.train()
    sum_loss = 0
    number_iterations = 0

    for x, attn_mask, y_mask, y in tqdm(data_loader, desc='train'):
        x = x.to(device)
        y = y.to(device)
        attn_mask = attn_mask.to(device)
        y = y.view(-1)

        y_predict = model(x, attn_mask)
        y_predict = y_predict.view(-1, y_predict.shape[2])
        loss = loss_fn(y_predict, y)
        sum_loss += loss.item()
        number_iterations += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum_loss/number_iterations

def fit(language, punctuation_type):
    device = get_device()
    print('device', device, torch.get_default_dtype())

    if SEED is not None:
        set_seed(SEED)
        print('seed', SEED)

    checkpoint_dir = join(NMODELS_DIR, punctuation_type, language)
    checkpoint = get_last_checkpoint(checkpoint_dir, 'epoch', is_need_max=True)
    checkpoint_epoch = get_checkpoint_epoch(checkpoint)
    print('checkpoint', checkpoint)

    model_options = PRETRAINED_MODELS[language]
    spacy_instance = SpacyNeuralNetwork()
    spacy_model = spacy_instance.get_model(language)
    spacy_special_tokens = get_special_tokens(spacy_model)
    spacy_instance.reset()
    tokenizer = get_tokenizer(model_options, spacy_special_tokens)
    punctuation_dictionary = get_punctuation_dictionary(punctuation_type)
    print('punctuation_dictionary', punctuation_type, punctuation_dictionary)
    print('tokenizer', len(tokenizer), get_nlp_all_tokens(tokenizer))

    model = DeepPunctuation(tokenizer, language, punctuation_type)

    if checkpoint:
        model.load_state_dict(torch.load(join(checkpoint_dir, checkpoint)))

    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        epoch_value = checkpoint_epoch + epoch + 1

        train_loader = torch.utils.data.DataLoader(
            Dataset(language, tokenizer, punctuation_type, is_train=True),
            num_workers=0,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            Dataset(language, tokenizer, punctuation_type, is_train=False),
            num_workers=0,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        train_loss = train(device, model, loss_fn, optimizer, train_loader)
        del train_loader

        val_loss, val_acc = validate(device, model, loss_fn, optimizer, val_loader)
        del val_loader

        model_name = f'model=epoch={epoch_value}=val_loss={val_loss:.8f}=val_acc={val_acc:.8f}=train_loss={train_loss:.8f}=name={punctuation_type}=language={language}=.pt'
        torch.save(model.state_dict(), join(checkpoint_dir, model_name))
        print(f'Result epoch: {epoch_value}/{str(checkpoint_epoch + EPOCHS)}; checkpoint: {model_name};')

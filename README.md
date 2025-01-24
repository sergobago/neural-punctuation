# Neural-punctuation
Project to train fast and accurate punctuation restoration neural network model for production on website.

## Prerequirements
* Python 3.9 for training

## Installation
`python -m venv venv; venv\Scripts\activate; pip install pipenv; pipenv install --dev;`

## Demo
https://tool-tube.com/punctuation

## Model architecture
1) Pre-trained model FacebookAI/xlm-roberta-base
2) Bi-LSTM layer
3) Linear layer

## Credits
Our article on [habr.ru](https://habr.ru)

Project is written based on code from repositories: [sviperm/neuro-comma](https://github.com/sviperm/neuro-comma),[xashru/punctuation-restoration](https://github.com/xashru/punctuation-restoration),[vlomme/Bert-Russian-punctuation](https://github.com/vlomme/Bert-Russian-punctuation)

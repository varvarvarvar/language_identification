import json
import numpy as np
from collections import MutableMapping

import mlflow
import torch
from sklearn.metrics import classification_report

import logging

from datagen import Dictionary, Encoder, pool_generator
from config import LOCAL_MODEL_STORAGE

logging.getLogger().setLevel(logging.INFO)


def predict_on_batch(model, data, batch_size, token_size, device):
    true, pred = [], []

    model.eval()

    with torch.no_grad():
        for batch in pool_generator(data, batch_size, token_size):
            # Get input and target sequences from batch
            X = [torch.from_numpy(d[0]) for d in batch]
            X_lengths = torch.tensor([x.numel() for x in X], dtype=torch.long, device='cpu')
            y = torch.tensor([d[1] for d in batch], dtype=torch.long, device=device)
            # Pad the input sequences to create a matrix
            X = torch.nn.utils.rnn.pad_sequence(X).to(device)
            pred_probs = model(X, X_lengths)
            pred_labels = torch.max(pred_probs, 1)[1]

            true += list(y.cpu().detach().numpy())
            pred += list(pred_labels.cpu().detach().numpy())

    return true, pred


def validate(model, data, batch_size, token_size, device, lang_vocab, epoch, tag=''):
    true, pred = predict_on_batch(model, data, batch_size, token_size, device)
    report = classification_report(
        true,
        pred,
        output_dict=True,
        target_names=lang_vocab.idx2token
        )
    report = flatten(report)
    report = {' '.join((tag, metric)): value for metric, value in report.items()}

    logging.info(f'| epoch {epoch:02d} | {str(report)}')
    mlflow.log_metrics(report, step=epoch)


class Predictor:
    def __init__(self, char_vocab_path, lang_vocab_path, params_path, model_path):

        self.char_vocab_path = char_vocab_path
        self.lang_vocab_path = lang_vocab_path
        self.params_path = params_path
        self.model_path = model_path

        self.__initialize__()
        self.__load_model__()

    def __initialize__(self):
        self.char_vocab = Dictionary().deserialize(self.char_vocab_path)
        self.lang_vocab = Dictionary().deserialize(self.lang_vocab_path)
        with open(self.params_path, 'r') as f:
            self.params = json.loads(f.read())

    def encode(self, texts):
        test_idxs, test_labels = Encoder().encode_unlabeled_data(texts, self.char_vocab)
        data = [(x, y) for x, y in zip(test_idxs, test_labels)]
        return data

    def __load_model__(self):
        self.model = mlflow.pytorch.load_model(
            self.model_path,
            map_location=torch.device('cpu'))

    def predict_on_encoded(self, data):
        idxs, pred = predict_on_batch(
            self.model,
            data,
            self.params['batch_size'],
            self.params['token_size'],
            device='cpu')

        idxs, pred = np.array(idxs), np.array(pred)
        order = np.argsort(idxs)

        pred = pred[order]
        pred_labels = np.array(self.lang_vocab.idx2token)[pred]

        return pred_labels

    def postprocess(self, pred_labels):
        mapper = {
            'dan': 'Danish',
            'swe': 'Swedish',
            'nno': 'Nynorsk',
            'nob': 'Bokmål',
            'other': 'Other'
        }
        return [mapper[label] for label in pred_labels]

    def predict(self, texts):
        try:
            encoded = self.encode(texts)
            pred_labels = self.predict_on_encoded(encoded)
            pred_labels = self.postprocess(pred_labels)

            return {'response': pred_labels}

        except Exception as e:
            logging.error(e)
            return {'response': None, 'error': str(e)}

    @classmethod
    def default_predictor(cls):

        char_vocab_path = './%s/char_vocab.json' % LOCAL_MODEL_STORAGE
        lang_vocab_path = './%s/lang_vocab.json' % LOCAL_MODEL_STORAGE
        model_path = './%s/model' % LOCAL_MODEL_STORAGE
        params_path = './%s/params.json' % LOCAL_MODEL_STORAGE

        try:
            predictor = cls(char_vocab_path, lang_vocab_path, params_path, model_path)
        except Exception as e:
            logging.warning('Model not found. Download model with serve/download_model.sh')
            logging.error(e)
            return {'response': None, 'error': str(e)}

        logging.info('Loaded model')
        return {'response': predictor}


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

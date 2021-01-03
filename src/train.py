import mlflow
import torch
from sklearn.metrics import classification_report

import logging

from collections import MutableMapping

from lstm import CharRNNClassifier
from datagen import pool_generator

logging.getLogger().setLevel(logging.INFO)


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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

  
def get_model(ntokens, embedding_size, hidden_size, nlabels, bidirectional, pad_index, device):
    model = CharRNNClassifier(ntokens, embedding_size, hidden_size, nlabels, bidirectional=bidirectional, pad_idx=pad_index).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    return model, optimizer


def validate(model, data, batch_size, token_size, device, lang_vocab, epoch, tag=''):
    true, pred = predict_on_batch(model, data, batch_size, token_size, device)
    report = classification_report(true, pred, output_dict=True, target_names=lang_vocab.idx2token)
    report = flatten(report)
    report = {' '.join((tag, metric)): value for metric, value in report.items()}

    logging.info(f'| epoch {epoch:02d} | {str(report)}')
    mlflow.log_metrics(report, step=epoch)


def train(model, optimizer, data, batch_size, token_size, criterion, device):
    model.train()

    total_loss, ncorrect, nsentences = 0, 0, 0

    for batch in pool_generator(data, batch_size, token_size, shuffle=True):
        # Get input and target sequences from batch
        X = [torch.from_numpy(d[0]) for d in batch]
        X_lengths = [x.numel() for x in X]
        X_lengths = torch.tensor(X_lengths, dtype=torch.long, device='cpu')
        y = torch.tensor([d[1] for d in batch], dtype=torch.long, device=device)
        # Pad the input sequences to create a matrix
        X = torch.nn.utils.rnn.pad_sequence(X).to(device)
        model.zero_grad()
        output = model(X, X_lengths)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        # Training statistics
        total_loss += loss.item()
        ncorrect += (torch.max(output, 1)[1] == y).sum().item()
        nsentences += y.numel()

    total_loss = total_loss / nsentences
    accuracy = 100 * ncorrect / nsentences

    return accuracy

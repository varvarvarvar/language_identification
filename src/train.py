import torch
from lstm import CharRNNClassifier
from datagen import pool_generator
from sklearn.metrics import classification_report
from collections import MutableMapping
import numpy as np
import mlflow


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

    # calculate accuracy on validation set

    with torch.no_grad():
        for batch in pool_generator(data, batch_size, token_size):
            # Get input and target sequences from batch
            X = [torch.from_numpy(d[0]) for d in batch]
            X_lengths = torch.tensor([x.numel() for x in X], dtype=torch.long, device=device)
            y = torch.tensor([d[1] for d in batch], dtype=torch.long, device=device)
            # Pad the input sequences to create a matrix
            X = torch.nn.utils.rnn.pad_sequence(X).to(device)
            answer = model(X, X_lengths)
            
            true += y
            pred += torch.max(answer, 1)[1]

    return true, pred

  
def get_model(ntokens, embedding_size, hidden_size, nlabels, bidirectional, pad_index, device):
    model = CharRNNClassifier(ntokens, embedding_size, hidden_size, nlabels, bidirectional=bidirectional, pad_idx=pad_index).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    return model, optimizer


def validate(model, data, batch_size, token_size, device):
    # calculate accuracy on validation set
    true, pred = predict_on_batch(model, data, batch_size, token_size, device)
    report = classification_report(true, pred, output_dict=True)
    report = flatten(report)
    mlflow.log_metrics(report)


def train(model, optimizer, data, batch_size, token_size, criterion, device):
    model.train()

    total_loss, ncorrect, nsentences = 0, 0, 0

    for batch in pool_generator(data, batch_size, token_size, shuffle=True):
        # Get input and target sequences from batch
        X = [torch.from_numpy(d[0]) for d in batch]
        X_lengths = [x.numel() for x in X]
        X_lengths = torch.tensor(X_lengths, dtype=torch.long, device=device)
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

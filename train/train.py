import torch

import logging

from lstm import CharRNNClassifier
from datagen import pool_generator

logging.getLogger().setLevel(logging.INFO)


def get_model(ntokens, embedding_size, hidden_size, nlabels, bidirectional, pad_index, device):
    model = CharRNNClassifier(
        ntokens,
        embedding_size,
        hidden_size,
        nlabels,
        bidirectional=bidirectional,
        pad_idx=pad_index).to(device
                              )
    optimizer = torch.optim.Adam(model.parameters())
    return model, optimizer


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

import random
import click

import torch
import mlflow
from sklearn.model_selection import train_test_split
import numpy as np

from seed import freeze_seed
from lstm import CharRNNClassifier
from datagen import Dictionary, batch_generator, pool_generator, encode_texts, encode_labels
from train import get_model, validate, train


@click.command()
@click.option("--epochs", type=click.INT, default=50, help="Number of train steps.")
@click.option("--batch_size", type=click.INT, default=256, help="Number of train steps.")
@click.option("--token_size", type=click.INT, default=200000, help="Number of train steps.")
@click.option("--hidden_size", type=click.INT, default=256, help="Number of train steps.")
@click.option("--embedding_size", type=click.INT, default=64, help="Number of train steps.")
def train_epochs(epochs, batch_size, token_size, hidden_size, embedding_size):

    # Read data

    x_train_full = open("../input/wili-2018/x_train.txt").read().splitlines()
    y_train_full = open("../input/wili-2018/y_train.txt").read().splitlines()

    x_test_full = open("../input/wili-2018/x_test.txt").read().splitlines()
    y_test_full = open("../input/wili-2018/y_test.txt").read().splitlines()

    x_train_full = x_train_full[:10]
    y_train_full = y_train_full[:10]
    x_test_full = x_test_full[:10]
    y_test_full = y_test_full[:10]

    # Get encoders

    char_vocab = Dictionary().char_dict(x_train_full)
    lang_vocab = Dictionary().lang_dict_scandi(y_train_full)

    # Convert data

    x_train_idx = encode_texts(char_vocab, x_train_full)
    y_train_idx = encode_labels(lang_vocab, y_train_full)

    x_test_idx = encode_texts(char_vocab, x_test_full)
    y_test_idx = encode_labels(lang_vocab, y_test_full)

    print(y_train_idx[0], x_train_idx[0][:10])

    x_train, x_val, y_train, y_val = train_test_split(x_train_idx, y_train_idx, test_size=0.15)

    train_data = [(x, y) for x, y in zip(x_train, y_train)]
    val_data = [(x, y) for x, y in zip(x_val, y_val)]
    test_data = [(x, y) for x, y in zip(x_test_idx, y_test_idx)]
    print(x_train[0])

    # mlflow.log_metrics({
    #     "train samples": len(train_data),
    #     "val samples": len(val_data)
    #     })

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Select 'GPU On' on kernel settings")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    bidirectional = False
    ntokens = len(char_vocab)
    nlabels = len(lang_vocab)
    pad_index = char_vocab.pad_index
    
    model, optimizer = get_model(ntokens, embedding_size, hidden_size, nlabels, bidirectional, pad_index, device)
    
    print(f'Training cross-validation model for {epochs} epochs')
    for epoch in range(epochs):
        train_acc = train(model, optimizer, train_data, batch_size, token_size, criterion, device)
        # print(f'| epoch {epoch:03d} | train accuracy={train_acc:.1f}%')
        validate(model, val_data, batch_size, token_size, device)
        validate(model, test_data, batch_size, token_size, device)
        # print(f'| epoch {epoch:03d} | val accuracy={valid_acc:.1f}%')

        # mlflow.log_metrics({
        #     "train_acc": train_acc,
        #     "val_acc": valid_acc,
        #     "test_acc": test_acc
        #     })

    # print(model)
    # for name, param in model.named_parameters():
    #     print(f'{name:20} {param.numel()} {list(param.shape)}')
    # print(f'TOTAL                {sum(p.numel() for p in model.parameters())}')
    
    # mlflow.pytorch.save_model(model, 'model')

if __name__ == '__main__':
    freeze_seed()
    train_epochs()

import mlflow
import torch
from sklearn.model_selection import train_test_split

import click
import logging

from seed import freeze_seed
from datagen import Dictionary, Encoder
from train import get_model, train
from validate import validate

logging.getLogger().setLevel(logging.INFO)


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

    x_train_full = x_train_full[:100]
    y_train_full = y_train_full[:100]
    x_test_full = x_test_full[:100]
    y_test_full = y_test_full[:100]

    # Get encoders

    char_vocab = Dictionary().char_dict(x_train_full)
    lang_vocab = Dictionary().lang_dict(y_train_full)

    # Convert data

    x_train_idx, y_train_idx = Encoder().encode_labeled_data(x_train_full, y_train_full, char_vocab, lang_vocab)
    x_test_idx, y_test_idx = Encoder().encode_labeled_data(x_test_full, y_test_full, char_vocab, lang_vocab)

    x_train, x_val, y_train, y_val = train_test_split(x_train_idx, y_train_idx, test_size=0.15)

    train_data = [(x, y) for x, y in zip(x_train, y_train)]
    val_data = [(x, y) for x, y in zip(x_val, y_val)]
    test_data = [(x, y) for x, y in zip(x_test_idx, y_test_idx)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        logging.warning("WARNING: CUDA is not available.")
    
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    bidirectional = False
    ntokens = len(char_vocab)
    nlabels = len(lang_vocab)
    pad_index = char_vocab.pad_index
    
    model, optimizer = get_model(ntokens, embedding_size, hidden_size, nlabels, bidirectional, pad_index, device)
       
    with mlflow.start_run():
        
        mlflow.log_metrics(
            {
                "train samples": len(train_data),
                "val samples": len(val_data),
                "test samples": len(test_data)
                }
            )

        mlflow.log_dict(lang_vocab.token2idx, "lang_vocab.json")
        mlflow.log_dict(char_vocab.token2idx, "char_vocab.json")
        params = {'epochs': epochs, 'batch_size': batch_size, 'token_size': token_size, 'hidden_size': hidden_size, 'embedding_size': embedding_size}
        mlflow.log_dict(params, "params.json")


        logging.info(f'Training cross-validation model for {epochs} epochs')
        
        for epoch in range(epochs):
            train_acc = train(model, optimizer, train_data, batch_size, token_size, criterion, device)
            logging.info(f'| epoch {epoch:02d} | train accuracy={train_acc:.1f}%')
            
            validate(model, val_data, batch_size, token_size, device, lang_vocab, tag='val', epoch=epoch)
            validate(model, test_data, batch_size, token_size, device, lang_vocab, tag='test', epoch=epoch)
            
            mlflow.pytorch.log_model(model, f'{epoch:02d}.model')

    mlflow.pytorch.log_model(model, 'model')


if __name__ == '__main__':
    freeze_seed()
    train_epochs()

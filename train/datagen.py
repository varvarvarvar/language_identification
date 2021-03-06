import json
import numpy as np
import random

import logging

logging.getLogger().setLevel(logging.INFO)


class Dictionary(object):
    def __init__(self):
        self.token2idx = {}
        self.idx2token = []

    def add_token(self, token):
        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        return self.token2idx[token]

    def __len__(self):
        return len(self.idx2token)

    @classmethod
    def char_dict(cls, texts):
        char_vocab = cls()

        pad_token = '<pad>'  # reserve index 0 for padding
        unk_token = '<unk>'  # reserve index 1 for unknown token

        cls.pad_index = char_vocab.add_token(pad_token)
        cls.unk_index = char_vocab.add_token(unk_token)

        chars = set(''.join(texts))
        for char in sorted(chars):
            char_vocab.add_token(char)

        logging.info(f"Vocabulary: {len(char_vocab)} UTF characters")

        return char_vocab

    @classmethod
    def lang_dict(cls, labels, target_languages={'dan', 'nor', 'swe', 'nno', 'nob'}):
        # use python set to obtain the list of languages without repetitions
        lang_vocab = cls()
        unk_token = 'other'
        cls.unk_index = lang_vocab.add_token(unk_token)
        for lang in set(labels):
            if lang in target_languages:
                lang_vocab.add_token(lang)

        logging.info(f"Labels: {len(lang_vocab)} languages")

        return lang_vocab

    @classmethod
    def deserialize(cls, path):

        vocab = cls()

        with open(path, 'r') as f:
            token2idx = json.loads(f.read())
        vocab.token2idx = token2idx

        idx2token = [None] * len(token2idx)
        for token, idx in token2idx.items():
            idx2token[idx] = token
        vocab.idx2token = idx2token

        return vocab


def batch_generator(data, batch_size, token_size):
    """Yield elements from data in chunks
    with a maximum of batch_size sequences
    and token_size tokens.
    """

    minibatch, sequences_so_far, tokens_so_far = [], 0, 0
    for ex in data:
        seq_len = len(ex[0])
        if seq_len > token_size:
            ex = (ex[0][:token_size], ex[1])
            seq_len = token_size
        minibatch.append(ex)
        sequences_so_far += 1
        tokens_so_far += seq_len
        if sequences_so_far == batch_size or tokens_so_far == token_size:
            yield minibatch
            minibatch, sequences_so_far, tokens_so_far = [], 0, 0
        elif sequences_so_far > batch_size or tokens_so_far > token_size:
            yield minibatch[:-1]
            minibatch, sequences_so_far, tokens_so_far = minibatch[-1:], 1, len(minibatch[-1][0])
    if minibatch:
        yield minibatch


def pool_generator(data, batch_size, token_size, shuffle=False):
    """Sort within buckets, then batch, then shuffle batches.
    Partitions data into chunks of size 100*token_size, sorts examples within
    each chunk, then batch these examples and shuffle the batches.
    """
    for p in batch_generator(data, batch_size * 100, token_size * 100):
        p_batch = batch_generator(
            sorted(p, key=lambda t: len(t[0]), reverse=True), batch_size, token_size
            )
        p_list = list(p_batch)
        if shuffle:
            for b in random.sample(p_list, len(p_list)):
                yield b
        else:
            for b in p_list:
                yield b


class Encoder():
    def __init__(self):
        pass

    def encode_texts(self, texts, char_vocab):
        x_test_idx = [
            np.array(
                [
                    char_vocab.token2idx[c]
                    if c in char_vocab.token2idx
                    else char_vocab.unk_index
                    for c in line
                    ]
                )
            for line in texts
            ]
        return x_test_idx

    def encode_labels(self, labels, lang_vocab):
        y_test_idx = np.array(
            [
                lang_vocab.token2idx[lang]
                if lang in lang_vocab.token2idx
                else lang_vocab.unk_index
                for lang in labels
                ]
            )
        return y_test_idx

    def encode_labeled_data(self, x_train, y_train, char_vocab, lang_vocab):
        x_train_idx = self.encode_texts(x_train, char_vocab)
        y_train_idx = self.encode_labels(y_train, lang_vocab)
        return x_train_idx, y_train_idx

    def encode_unlabeled_data(self, texts, char_vocab):
        x_idx = self.encode_texts(texts, char_vocab)
        y_idx = [i for i in range(len(texts))]
        return x_idx, y_idx

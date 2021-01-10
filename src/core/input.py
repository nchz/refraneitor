import math
import random

import fasttext
import numpy as np
import tensorflow as tf


BEGIN_TOKEN = "!begin"
END_TOKEN = "!end"


class BatchGenerator(tf.keras.utils.Sequence):
    """Class to handle model input."""

    def __init__(
        self,
        raw_data_path,
        fasttext_model_path,
        batch_size,
        shuffle=True,
    ):
        self.raw_data_path = raw_data_path
        self.fasttext_model_path = fasttext_model_path
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.get_raw_data()
        self.get_vocabulary()
        self.build_dataset()

        self.len = math.ceil(len(self.dataset) / batch_size)

    def get_raw_data(self):
        """Load text from raw_data_path and pad sequences."""
        # load sentences in a list of lists.
        with open(self.raw_data_path) as f:
            self.raw_data = [[BEGIN_TOKEN] + sentence.split() + [END_TOKEN] for sentence in f.readlines()]

        # length of a sentence == number of words.
        self.maxlen = max(len(ws) for ws in self.raw_data)

        # pad sequences. fasttext returns an all-zero vector for the empty string:
        # ft.get_word_vector("") == np.zeros(ft.get_dimension())
        pad = [""] * self.maxlen
        # pre-padding.
        self.raw_data = [(ws + pad)[: self.maxlen] for ws in self.raw_data]
        # post-padding.
        # self.raw_data = [(pad + ws)[-self.maxlen :] for ws in self.raw_data]

    def get_vocabulary(self):
        """Build dictionaries for vocabulary."""
        self.vocab = sorted(set(w for ws in self.raw_data for w in ws))
        self.id2word = dict(enumerate(self.vocab))
        self.word2id = {v: k for k, v in self.id2word.items()}
        # note: self.id2word[0] == ""

    def build_dataset(self):
        """Build vectors and labels from raw data. Intended to be called just once."""
        # load fasttext model.
        ft = fasttext.load_model(self.fasttext_model_path)

        # get input vector for each word in each sentence.
        xs = np.array([[ft.get_word_vector(w) for w in ws] for ws in self.raw_data])
        del ft

        # get one-hot labels for each timestep.
        ys = np.zeros(shape=(len(self.raw_data), self.maxlen, len(self.vocab)))
        for i, ws in enumerate(self.raw_data):
            # skip first token for each sentence.
            for j, w in enumerate(ws[1:]):
                ys[i, j, self.word2id[w]] = 1

        self.dataset = list(zip(xs, ys))
        if self.shuffle:
            random.shuffle(self.dataset)

    # methods required by tf.keras.utils.Sequence

    def __len__(self):
        return self.len

    def __getitem__(self, batch_index):
        batch_index %= self.len  # tf.keras.utils.Sequence may be iterated infinitely.
        i = self.batch_size * batch_index
        xs, ys = zip(*self.dataset[i : i + self.batch_size])
        return np.array(xs), np.array(ys)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.dataset)

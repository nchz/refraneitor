import random

import fasttext
import numpy as np

from core.model import Refraneitor


def generate_sequence(
    rnn_units,
    bg,
    initial_seq,
    weights_to_load,
    vocab_size,
    mode='prob',
):
    ft = fasttext.load_model("/fasttext-models/cc.es.300.bin")

    model = Refraneitor(
        rnn_units=rnn_units,
        max_seq_length=bg.maxlen,
        word_embedding_dim=bg.dataset[0][0].shape[1],
        vocab_size=len(bg.vocab),
        is_training=False,
    )

    model.load_weights(weights_to_load)

    model.summary()

    seq = [ft.get_word_vector(w) for w in initial_seq]

    # generate sequence.
    sample = None
    while sample != 0:
        predictions = model.predict_on_batch(np.array([seq])).ravel()

        if mode == 'prob':
            sample = np.random.choice(range(vocab_size), p=predictions)
        elif mode == 'highest':
            sample = np.argmax(predictions)

        next_word = bg.id2word[sample]
        seq.append(ft.get_word_vector(next_word))

    return initial_seq

import tensorflow as tf

from core.input import BatchGenerator
from core.model import Refraneitor

def train_model(
    batch_size,
    rnn_units,
    epochs,
    raw_data_path="/data/dataset.txt",
    fasttext_model_path="/fasttext-models/cc.es.300.bin",
    initial_epoch=0,
):
    bg = BatchGenerator(
        raw_data_path=raw_data_path,
        fasttext_model_path=fasttext_model_path,
        batch_size=batch_size,
    )

    refraneitor = Refraneitor(
        rnn_units=rnn_units,
        max_seq_length=bg.maxlen,
        word_embedding_dim=bg.dataset[0][0].shape[1],
        vocab_size=len(bg.vocab),
    )

    refraneitor.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="/data/dumps/weights-{epoch:02}.h5",
            monitor="acc",
            # save_best_only=True,
            save_weights_only=True,
            period=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename="/data/logs/log.csv",
            append=(initial_epoch != 0),  # append only if resuming training.
        ),
    ]

    # train the model.
    history = refraneitor.fit_generator(
        bg,
        shuffle=False,  # already handled by `BatchGenerator.on_epoch_end` method.
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1,
        # use_multiprocessing=True
    )

    return bg, refraneitor  # history

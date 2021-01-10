import tensorflow as tf


RNNLayer = tf.keras.layers.LSTM  # to easily change RNN implementation (GRU. LSTM, etc).


class Refraneitor(tf.keras.models.Sequential):
    """Model architecture."""

    def __init__(
        self,
        max_seq_length,
        word_embedding_dim,
        rnn_units,
        vocab_size,
        is_training=True,
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        **kwargs
    ):
        super().__init__()

        if is_training:
            self.add(
                tf.keras.Input(
                    shape=(max_seq_length, word_embedding_dim)
                )
            )
        else:
            self.add(
                tf.keras.Input(
                    shape=(None, word_embedding_dim)
                )
            )

        # # project input word embeddings to a new space.
        # self.add(tf.keras.layers.Dense(128))

        for units in rnn_units[:-1]:
            self.add(
                RNNLayer(
                    units=units,
                    return_sequences=True,  # when stacking RNN layers so shapes match.
                    **kwargs
                )
            )

        # last RNN layer may have a different return_sequences value.
        self.add(
            RNNLayer(
                units=rnn_units[-1],
                return_sequences=is_training,
                **kwargs
            )
        )

        # when training, wrap last layer to apply the same weights in all timesteps.
        if is_training:
            self.add(
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(
                        vocab_size,
                        activation="softmax",
                    )
                )
            )
        else:
            self.add(
                tf.keras.layers.Dense(
                    vocab_size,
                    activation="softmax",
                )
            )

        self.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )

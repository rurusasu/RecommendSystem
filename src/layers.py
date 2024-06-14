from typing import List

import tensorflow as tf


class LightSE(tf.keras.Model):
    def __init__(self, field_size, embedding_size=32):
        super(LightSE, self).__init__()
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.excitation = tf.keras.Sequential(
            [tf.keras.layers.Dense(self.field_size, use_bias=False)]
        )
        self.softmax = tf.keras.layers.Softmax(axis=1)

    def call(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                f"Unexpected inputs dimensions {len(inputs.shape)}, expect to be 3 dimensions"
            )

        Z = tf.reduce_mean(inputs, axis=-1)
        A = self.excitation(Z)
        A = self.softmax(A)
        out = inputs * tf.expand_dims(A, axis=2)

        return inputs + out

    def get_config(self):
        config = super(LightSE, self).get_config()
        config.update(
            {
                "field_size": self.field_size,
                "embedding_size": self.embedding_size,
            }
        )
        return config


class DNN(tf.keras.Model):
    def __init__(
        self,
        layer_sizes: List[int],
        activation="relu",
        use_bn: bool = False,
        use_ln: bool = False,
    ):
        super(DNN, self).__init__()
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.dense_layers = tf.keras.Sequential()

        for layer_size in layer_sizes:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))
            if self.use_bn:
                self.dense_layers.add(tf.keras.layers.BatchNormalization())
            elif use_ln:
                self.dense_layers.add(tf.keras.layers.LayerNormalization())
            self.dense_layers.add(tf.keras.layers.Activation(activation))

        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense_layers(inputs)
        return self.output_layer(x)

    def get_config(self):
        config = super(DNN, self).get_config()
        config.update(
            {
                "layer_sizes": self.layer_sizes,
                "activation": self.activation,
                "use_bn": self.use_bn,
                "use_ln": self.use_ln,
            }
        )
        return config

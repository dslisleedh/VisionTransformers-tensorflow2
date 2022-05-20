import tensorflow as tf
import numpy as np
from einops import rearrange
import math


class Droppath(tf.keras.layers.Layer):
    def __init__(self, survivla_prob):
        super(Droppath, self).__init__()
        self.survivla_prob = survivla_prob

    @tf.function
    def call(self, inputs, **kwargs):
        if self.survivla_prob == 1.:
            return inputs
        if self.survivla_prob == 0.:
            return tf.zeros_like(inputs)

        if tf.keras.backend.learning_phase():
            epsilon = tf.keras.backend.random_bernoulli(shape=[tf.shape(inputs)[0]] + [1 for _ in range(len(tf.shape(inputs)) - 1)],
                                                        p=self.survivla_prob,
                                                        dtype='float32'
                                                        )
            return inputs * epsilon
        else:
            return inputs * self.survivla_prob


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 n_patches
                 ):
        super(PositionalEncoding, self).__init__()
        self.n_filters = n_filters
        self.n_patches = n_patches

        pe = tf.Variable(
            initial_value=tf.concat([
                self.positional_encoding(i) for i in range(self.n_patches)
            ], axis=0),
            trainable=False
        )
        self.pe = tf.cast(pe, dtype=tf.float32)

    def positional_encoding(self, pos):
        pe = np.zeros(self.n_filters)
        for i in range(0, self.n_filters, 2):
            pe[i] = np.math.sin(pos / (10000 ** ((2 * i) / self.n_filters)))
            pe[i + 1] = np.math.cos(pos / (10000 ** ((2 * i) / self.n_filters)))
        return np.expand_dims(pe, 0)

    def call(self, inputs, *args, **kwargs):
        return inputs + self.pe


class ChannelMLP(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 expansion_rate=4
                 ):
        super(ChannelMLP, self).__init__()
        self.n_filters = n_filters
        self.expansion_rate = expansion_rate

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Dense(self.n_filters * self.expansion_rate,
                                  activation='gelu'
                                  ),
            tf.keras.layers.Dense(self.n_filters)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 n_heads
                 ):
        super(ScaledDotProductAttention, self).__init__()
        self.n_filters = n_filters
        self.n_heads = n_heads
        self.scale = tf.sqrt(self.n_filters / self.n_heads)

        self.to_qkv = tf.keras.layers.Dense(self.n_filters * 3,
                                            activation='linear'
                                            )

    def to_heads(self, x):
        return rearrange(x, 'b p (h c) -> b h p c',
                         h=self.n_heads
                         )

    def call(self, inputs, *args, **kwargs):
        q, k, v = tf.split(self.to_qkv(inputs),
                           num_or_size_splits=3,
                           axis=-1
                           )
        q, k, v = [self.to_heads(qkv) for qkv in [q, k, v]]

        attention_map = tf.matmul(q, k,
                                  transpose_b=True
                                  )
        attention_map = attention_map / self.scale
        attention_map = tf.nn.softmax(attention_map, axis=-1)

        v = tf.matmul(attention_map, v)
        return rearrange(v, 'b h p c -> b p (h c)')


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 n_heads,
                 survival_prob
                 ):
        super(EncoderBlock, self).__init__()
        self.n_filters = n_filters
        self.n_heads = n_heads
        self.survival_prob = survival_prob

        self.ln1 = tf.keras.layers.LayerNormalization()
        self.spatial = ScaledDotProductAttention(self.n_filters,
                                                 self.n_heads
                                                 )
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.channel = ChannelMLP(self.n_filters)
        self.droppath = Droppath(self.survival_prob)

    def call(self, inputs, *args, **kwargs):
        inputs = self.droppath(self.spatial(self.ln1(inputs))) + inputs
        inputs = self.droppath(self.channel(self.ln2(inputs))) + inputs
        return inputs

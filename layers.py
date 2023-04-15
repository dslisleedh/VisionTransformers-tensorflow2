import tensorflow as tf
from einops import rearrange

import numpy as np

from typing import Sequence, Optional


class DropPath(tf.keras.layers.Layer):
    def __init__(
            self, module: tf.keras.layers.Layer, survival_prob: float = 0., **kwargs
    ):
        super(DropPath, self).__init__(**kwargs)
        self.module = module
        self.survival_prob = survival_prob

    def call(
            self, inputs: tf.Tensor, training: Optional[bool] = None,
            *args, **kwargs
    ) -> tf.Tensor:
        training = training or False

        if self.survival_prob == 1. or not training:
            return self.module(inputs, *args, **kwargs)

        else:
            if self.survival_prob == 0.:
                return inputs
            else:
                survival_state = tf.random.uniform(
                    shape=(), minval=0., maxval=1., dtype=tf.float32
                ) < self.survival_prob

                output = tf.cond(
                    survival_state,
                    lambda: inputs + ((self.module(inputs) - inputs) / tf.cast(self.survival_prob, tf.float32)),
                    lambda: inputs
                )
                return output


class LinearProj(tf.keras.layers.Layer):
    def __init__(self, n_filters: int, patch_size: Sequence[int], **kwargs):
        super(LinearProj, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.patch_size = patch_size

    def build(self, input_shape: tf.TensorShape):
        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.n_filters, self.patch_size, strides=self.patch_size, padding='VALID'
            ),
            tf.keras.layers.Reshape((-1, self.n_filters))
        ])

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return self.forward(inputs)


class CLSToken(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CLSToken, self).__init__(**kwargs)

    def build(self, input_shape: tf.TensorShape):
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, input_shape[-1]),
            initializer=tf.keras.initializers.truncated_normal(stddev=0.02),
            trainable=True
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        shape = tf.shape(inputs)
        cls_token = tf.broadcast_to(self.cls_token, (shape[0], 1, shape[-1]))
        return tf.concat([cls_token, inputs], axis=1)


class PosEnc(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PosEnc, self).__init__(**kwargs)

    def build(self, input_shape: tf.TensorShape):
        self.pos_enc = self.add_weight(
            name='pos_enc',
            shape=(1, input_shape[1], input_shape[-1]),
            initializer=tf.keras.initializers.truncated_normal(stddev=0.02),
            trainable=True
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return inputs + self.pos_enc


class MHSA(tf.keras.layers.Layer):
    def __init__(self, n_heads: int, **kwargs):
        super(MHSA, self).__init__(**kwargs)
        self.n_heads = n_heads

    def build(self, input_shape: tf.TensorShape):
        self.to_qkv = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(input_shape[-1] * 3, use_bias=False)
        ])
        self.to_out = tf.keras.layers.Dense(input_shape[-1])

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        qkv = self.to_qkv(inputs)
        q, k, v = tf.split(qkv, 3, axis=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.n_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.n_heads)

        score = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(k.shape[-1], tf.float32))
        attn = tf.nn.softmax(score, axis=-1)
        out = tf.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class MLP(tf.keras.layers.Layer):
    def __init__(self, exp_ratio: int, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.exp_ratio = exp_ratio

    def build(self, input_shape: tf.TensorShape):
        self.forward = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(input_shape[-1] * self.exp_ratio, activation='gelu'),
            tf.keras.layers.Dense(input_shape[-1])
        ])

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return self.forward(inputs)


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, n_heads: int, exp_ratio: int, drop_probs: np.ndarray, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.exp_ratio = exp_ratio
        self.drop_probs = drop_probs

    def build(self, input_shape: tf.TensorShape):
        self.mhsa = DropPath(MHSA(self.n_heads), self.drop_probs[0])
        self.mlp = DropPath(MLP(self.exp_ratio), self.drop_probs[1])

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return self.mlp(self.mhsa(inputs))

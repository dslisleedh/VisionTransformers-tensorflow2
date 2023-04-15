import numpy as np

from layers import *
import tensorflow as tf


class VisionTransformer(tf.keras.models.Model):
    def __init__(
            self, n_filters: int, patch_size: Sequence[int], n_blocks: int,
            n_heads: int, exp_ratio: int, drop_rates: float, n_classes: int,
            **kwargs
    ):
        super(VisionTransformer, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.patch_size = patch_size
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.exp_ratio = exp_ratio
        self.drop_rates = drop_rates
        self.n_classes = n_classes

    def build(self, input_shape: tf.TensorShape):
        self.proj = tf.keras.Sequential([
            LinearProj(self.n_filters, self.patch_size),
            CLSToken(),
            PosEnc()
        ])
        survival_probs = 1. - np.linspace(0., self.drop_rates, self.n_blocks * 2).reshape(self.n_blocks, 2)
        self.blocks = tf.keras.Sequential([
            EncoderBlock(self.n_heads, self.exp_ratio, survival_probs[i]) for i in range(self.n_blocks)
        ])
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.gather(x, 0, axis=1)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(self.n_classes, activation='softmax', kernel_initializer='zeros')
        ])

    @tf.function(jit_compile=True)
    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        feats = self.proj(inputs)
        feats = self.blocks(feats)
        return self.classifier(feats)


if __name__ == '__main__':
    # For testing if jit compile works
    model = VisionTransformer(512, (16, 16), 12, 8, 4, 0.1, 10)

    inputs = tf.random.normal((1, 224, 224, 3))

    outputs = model(inputs)
    print(outputs.shape)

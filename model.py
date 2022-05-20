from layers import *
import tensorflow as tf


class VisionTransformer(tf.keras.models.Model):
    def __init__(self,
                 n_filters,
                 input_res,
                 patch_res,
                 n_layers,
                 stochastic_depth_rate,
                 n_labels = 1000,
                 n_heads=8
                 ):
        super(VisionTransformer, self).__init__()
        self.n_filters = n_filters
        self.n_heads = n_heads
        self.input_res = input_res
        self.patch_res = patch_res
        self.n_patches = 1 + (input_res // patch_res) ** 2
        self.n_layers = n_layers
        self.n_labels = n_labels
        self.stochastic_depth_rate = stochastic_depth_rate
        self.survival_prob = 1. - tf.linspace(0., self.stochastic_depth_rate, self.n_layers)

        self.patch_embedding = tf.keras.layers.Conv2D(self.n_filters,
                                                      kernel_size=self.patch_res,
                                                      strides=self.patch_res,
                                                      padding='VALID',
                                                      activation='linear',
                                                      use_bias=False
                                                      )
        self.pos_embedding = PositionalEncoding(self.n_filters, self.n_patches)
        self.encoder_blocks = tf.keras.Sequential([
            EncoderBlock(self.n_filters,
                         self.n_heads,
                         survival_prob
                         ) for survival_prob in self.survival_prob
        ])
        self.classifier = tf.keras.layers.Dense(self.n_labels,
                                                activation='softmax',
                                                kernel_initializer=tf.keras.initializers.zeros()
                                                )

    def forward(self, x):
        shape = x.shape
        embedding = tf.concat([
            tf.zeros((shape[0], 1, self.n_filters)),
            self.pos_embedding(self.patch_embedding(x))
        ], axis=0)
        cls_token = self.encoder_blocks(embedding)[:, 0, :]
        y_hat = self.classifier(cls_token)
        return y_hat

    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)

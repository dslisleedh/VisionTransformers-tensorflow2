import tensorflow as tf


class MLP(tf.keras.layers.Layer):
  def __init__(self,
               projection_dim,
               dropout_rate
               ):
    super(MLP, self).__init__()
    self.projection_dim = projection_dim
    self.dropout_rate = dropout_rate

    self.forward = tf.keras.Sequential([
      tf.keras.layers.Dense(self.projection_dim*2,
                            activation='gelu'
                            ),
      tf.keras.layers.Dropout(self.dropout_rate),
      tf.keras.layers.Dense(self.projection_dim,
                            activation='gelu'
                            ),
      tf.keras.layers.Dropout(self.dropout_rate)
    ])

    def call(self, inputs, *args, **kwargs):
      return self.forward(inputs)


class TransformerEncoder(tf.keras.layers.Layer):
  def __init__(self,
               num_heads,
               projection_dim,
               mlp_dims,
               dropout_rate
               ):
    super(TransformerEncoder, self).__init__()
    self.num_heads = num_heads
    self.projection_dim = projection_dim
    self.mlp_dims = mlp_dims
    self.dropout_rate = dropout_rate

    self.LN1 = tf.keras.layers.LayerNormalization()
    self.MHA = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,
                                                  key_dim=self.projection_dim,
                                                  dropout_rate=self.dropout_rate
                                                  )
    self.LN2 = tf.keras.layers.LayerNormalization()
    self.MLP = MLP(self.projection_dim, self.dropout_rate)

  def call(self, inputs, *args, **kwargs):
    residual = self.LN1(inputs)
    residual = self.MHA(residual, residual)
    inputs = inputs + residual
    residual = self.LN2(inputs)
    residual = self.MLP(residual)
    inputs = inputs + residual
    return inputs


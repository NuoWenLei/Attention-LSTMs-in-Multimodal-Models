from imports import tf

class ResidualMultiHeadAttentionUnit(tf.keras.layers.Layer):

	def __init__(self,
	num_heads: int,
	output_size: int,
	input_shape: tuple,
	residual: bool,
	name: str):
		super().__init__(name = name)
		self.num_heads = num_heads
		self.output_size = output_size
		self.input_shape_manual = input_shape
		self.residual = residual

		self.mha = tf.keras.layers.MultiHeadAttention(
			num_heads = self.num_heads,
			key_dim = self.output_size
		)

		self.out_dense = tf.keras.layers.Dense(self.output_size, activation = "relu")

		if self.residual:
			self.layer_norm = tf.keras.layers.LayerNormalization()
	
	def call(self, inputs):
		x = self.mha(inputs)

		if self.residual:
			x = self.layer_norm(inputs + x)
		
		x = self.out_dense(x)
		return x

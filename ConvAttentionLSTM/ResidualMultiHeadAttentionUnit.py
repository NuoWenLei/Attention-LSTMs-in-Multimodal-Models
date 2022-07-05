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
			self.conv_residual = tf.keras.layers.Conv1D(self.output_size, 1)
	
	def call(self, inputs):
		x = self.mha(inputs, inputs)

		if self.residual:
			if tf.shape(inputs)[-2] != tf.shape(x)[-2]:
				input_features = self.conv_residual(tf.reshape(inputs, [*tf.shape(inputs)[:-2], -1]))
				x = self.layer_norm(x + input_features)
			else:
				x = self.layer_norm(x + inputs)
		
		x = self.out_dense(x)
		return x

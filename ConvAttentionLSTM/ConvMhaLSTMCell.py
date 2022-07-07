from imports import tf
from ConvMultiHeadAttentionUnit import ConvMultiHeadAttentionUnit

class ConvMhaLSTMCell(tf.keras.layers.Layer):

	def __init__(self,
	units: int,
	num_heads: int,
	d_model: int,
	output_size: int,
	image_dims: tuple,
	name: str,
	activation = tf.keras.activations.tanh,
	recurrent_activation = tf.keras.activations.hard_sigmoid,
	attention_type: str = "local_1d",
	mha_feature_activation: str = "relu",
	mha_output_activation: str = "linear",
	query_block_length = None
	):
		super().__init__(name = name)
		self.units = units
		self.num_heads = num_heads
		self.d_model = d_model
		self.output_size = output_size
		self.image_dims = image_dims
		self.activation = activation
		self.recurrent_activation = recurrent_activation
		self.attention_type = attention_type
		self.mha_feature_activation = mha_feature_activation
		self.mha_output_activation = mha_output_activation
		self.query_block_length = query_block_length
		self.state_size = [
			tf.TensorShape([self.image_dims[0] * self.image_dims[1], self.output_size]),
			tf.TensorShape([self.image_dims[0] * self.image_dims[1], self.output_size])]
		
		(self.input_attention_i, self.input_attention_f,
		self.input_attention_o, self.input_attention_c) = [
			ConvMultiHeadAttentionUnit(
				num_heads = self.num_heads,
				d_model = self.d_model,
				output_size = self.output_size,
				name = f"{name}_InputAttention_{i}",
				feature_activation = self.mha_feature_activation,
				output_activation = self.mha_output_activation,
				attention_type = self.attention_type,
				query_block_length = self.query_block_length
			) for i in range(4)]

		(self.recurrent_attention_i, self.recurrent_attention_f,
		self.recurrent_attention_o, self.recurrent_attention_c) = [
			ConvMultiHeadAttentionUnit(
				num_heads = self.num_heads,
				d_model = self.d_model,
				output_size = self.output_size,
				name = f"{name}_InputAttention_{i}",
				feature_activation = self.mha_feature_activation,
				output_activation = self.mha_output_activation,
				attention_type = self.attention_type,
				query_block_length = self.query_block_length
			) for i in range(4)]

	def call(self, inputs, states):

		prev_h = states[0]
		prev_c = states[1]

		# Processing inputs
		x_i = self.input_attention_i(inputs)
		x_f = self.input_attention_f(inputs)
		x_c = self.input_attention_c(inputs)
		x_o = self.input_attention_o(inputs)

		# Processing hidden state
		h_i = self.recurrent_attention_i(prev_h, recurrent = True)
		h_f = self.recurrent_attention_f(prev_h, recurrent = True)
		h_c = self.recurrent_attention_c(prev_h, recurrent = True)
		h_o = self.recurrent_attention_o(prev_h, recurrent = True)

		# LSTM calculations
		i = tf.cast(self.recurrent_activation(x_i + h_i), tf.float32)
		f = tf.cast(self.recurrent_activation(x_f + h_f), tf.float32)
		g = tf.cast(self.activation(x_c + h_c), tf.float32)
		o = tf.cast(self.recurrent_activation(x_o + h_o), tf.float32)
		c = f * prev_c + i * g
		h = o * self.activation(c)

		return h, [h, c]
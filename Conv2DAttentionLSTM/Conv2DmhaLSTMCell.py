# See https://github.com/keras-team/keras/blob/v2.9.0/keras/layers/rnn/base_conv_lstm.py#L197
# to create MultiHeadGraphAttentionLSTMCell layer
#
# See https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/layers/rnn/base_conv_lstm.py#L277
# for changes needed in base function for recurrent compatibility

from imports import tf, Iterable
from Conv2DMHAUnit import Conv2DMHAUnit

class Conv2DmhaLSTMCell(tf.keras.layers.Layer):

	def __init__(self,
	units: int,
	num_heads: int,
	d_model: int,
	image_dims: tuple,
	kernel_size: Iterable,
	name: str,
	activation = tf.keras.activations.tanh,
	recurrent_activation = tf.keras.activations.hard_sigmoid,
	mha_feature_activation: str = "relu",
	mha_output_activation: str = "linear"
	):
		super().__init__(name = name)
		self.units = units
		self.num_heads = num_heads
		self.d_model = d_model
		self.image_dims = image_dims
		self.kernel_size = kernel_size
		self.activation = activation
		self.recurrent_activation = recurrent_activation
		self.mha_feature_activation = mha_feature_activation
		self.mha_output_activation = mha_output_activation
		self.state_size = [
			tf.TensorShape([self.image_dims[0], self.image_dims[1], self.d_model]),
			tf.TensorShape([self.image_dims[0], self.image_dims[1], self.d_model])]
		
		(self.input_attention_i, self.input_attention_f,
		self.input_attention_o, self.input_attention_c) = [
			Conv2DMHAUnit(
				num_heads = self.num_heads,
				d_model = self.d_model,
				image_size = self.image_dims,
				kernel_size = self.kernel_size,
				name = f"{name}_InputAttention_{i}",
				feature_activation = self.mha_feature_activation,
				output_activation = self.mha_output_activation
			) for i in range(4)]

		(self.recurrent_attention_i, self.recurrent_attention_f,
		self.recurrent_attention_o, self.recurrent_attention_c) = [
			Conv2DMHAUnit(
				num_heads = self.num_heads,
				d_model = self.d_model,
				image_size = self.image_dims,
				kernel_size = self.kernel_size,
				name = f"{name}_RecurrentAttention_{i}",
				feature_activation = self.mha_feature_activation,
				output_activation = self.mha_output_activation
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
		h_i = self.recurrent_attention_i(prev_h)
		h_f = self.recurrent_attention_f(prev_h)
		h_c = self.recurrent_attention_c(prev_h)
		h_o = self.recurrent_attention_o(prev_h)

		# LSTM calculations
		i = tf.cast(self.recurrent_activation(x_i + h_i), tf.float32)
		f = tf.cast(self.recurrent_activation(x_f + h_f), tf.float32)
		g = tf.cast(self.activation(x_c + h_c), tf.float32)
		o = tf.cast(self.recurrent_activation(x_o + h_o), tf.float32)
		c = f * prev_c + i * g
		h = o * self.activation(c)

		return h, [h, c]
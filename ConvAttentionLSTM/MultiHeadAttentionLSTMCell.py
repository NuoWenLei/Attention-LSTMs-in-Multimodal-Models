from imports import tf

class MultiHeadAttentionLSTMCell(tf.keras.models.Model):
	
	def __init__(self,
			units: int,
			num_heads: int,
			sequence_length: int,
			output_size: int,
			input_shape: tuple,
			residual: bool,
			name: str,
			activation = tf.keras.activations.tanh,
			recurrent_activation = tf.keras.activations.hard_sigmoid):
		super().__init__(name = name)
		self.units = units
		self.num_heads = num_heads
		self.seq_len = sequence_length
		self.output_size = output_size
		self.state_size = self.output_size
		self.input_shape = input_shape
		self.residual = residual
		self.activation = activation
		self.recurrent_activation = recurrent_activation

		(self.input_attention_i, self.input_attention_f,
		self.input_attention_o, self.input_attention_c) = [
			self.residual_multi_head_attention(
				recurrent = False,
				name = f"{name}_InputAttention_{i}"
			) for i in range(4)]
		
		(self.recurrent_attention_i, self.recurrent_attention_f,
		self.recurrent_attention_o, self.recurrent_attention_c) = [
			self.residual_multi_head_attention(
				recurrent = True,
				name = f"{name}_RecurrentAttention_{i}"
				) for i in range(4)]

	def residual_multi_head_attention(
		self,
		name: str,
		recurrent: bool = False):

		if recurrent:
			inp = tf.keras.layers.Input(shape = (self.output_size,))
		else:
			inp = tf.keras.layers.Input(shape = self.input_shape)
		res = tf.keras.layers.MultiHeadAttention(
			num_heads = self.num_heads,
			key_dim = self.output_size
		)(inp, inp)

		if self.residual:
			res = tf.keras.layers.LayerNormalization(inp + res)
		
		out = tf.keras.layers.Dense(self.output_size, activation = "relu")(res)
		return tf.keras.models.Model(inputs = inp, outputs = out, name = name)

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


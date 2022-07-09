from imports import tf

class MultiHeadAttentionLSTMCell(tf.keras.layers.Layer):
	
	def __init__(self,
			units: int,
			num_heads: int,
			d_model: int,
			num_tokens: int,
			name: str,
			activation = tf.keras.activations.tanh,
			recurrent_activation = tf.keras.activations.hard_sigmoid):
		super().__init__(name = name)
		self.units = units
		self.num_heads = num_heads
		self.d_model = d_model
		self.num_tokens = num_tokens
		self.state_size = [tf.TensorShape([self.num_tokens, self.d_model]), tf.TensorShape([self.num_tokens, self.d_model])]
		self.activation = activation
		self.recurrent_activation = recurrent_activation

		(self.input_attention_i, self.input_attention_f,
		self.input_attention_o, self.input_attention_c) = [
			tf.keras.layers.MultiHeadAttention(
				num_heads = self.num_heads,
				key_dim = self.d_model,
				name = f"{name}_InputAttention_{i}"
			) for i in range(4)]
		
		(self.recurrent_attention_i, self.recurrent_attention_f,
		self.recurrent_attention_o, self.recurrent_attention_c) = [
			tf.keras.layers.MultiHeadAttention(
				num_heads = self.num_heads,
				key_dim = self.d_model,
				name = f"{name}_RecurrentAttention_{i}"
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


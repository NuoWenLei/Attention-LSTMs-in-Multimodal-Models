from imports import tf

class AttentionBottleneckUnit(tf.keras.layers.Layer):

	def __init__(self,
	num_pad_tokens: int,
	num_heads: int,
	d_model_1: int,
	d_model_2: int,
	name: str):
		super().__init__(name = name)
		self.num_pad_tokens = num_pad_tokens
		self.num_heads = num_heads
		self.d_model_1 = d_model_1
		self.d_model_2 = d_model_2
		self.mha = tf.keras.layers.MultiHeadAttention(
			self.num_heads,
			self.d_model_1 + self.d_model_2 + self.num_pad_tokens)

	def call(self, inp1, inp2):
		b = tf.shape(inp1)[0]
		inp1 = tf.reshape(inp1, (b, -1))
		inp2 = tf.reshape(inp2, (b, -1))
		mha_input = tf.concat([inp1, tf.zeros((b, self.num_pad_tokens)), inp2], axis = 1)
		bottleneck_attention = self.mha(mha_input, mha_input)
		return bottleneck_attention[:, :self.d_model_1], bottleneck_attention[:, self.d_model_1 + self.num_pad_tokens:]

	# TODO: Decide whether to make a model that:
	# - has bottleneck attention between units in the LSTM
	# - has bottleneck attention between LSTM layers

	
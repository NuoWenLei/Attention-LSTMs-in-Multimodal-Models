from imports import tf

class ConvMultiHeadAttentionUnit(tf.keras.layers.Layer):

	def __init__(self,
	num_heads: int,
	d_model: int,
	output_size: int,
	name: str,
	feature_activation: str = "relu",
	output_activation: str = "linear",
	attention_type: str = "local_1d",
	query_block_length = None,
	query_kernel_size = None):

		super().__init__(name = name)

		self.num_heads = num_heads
		self.d_model = d_model
		self.query_size = self.d_model // self.num_heads
		self.feature_activation = feature_activation
		self.output_activation = output_activation
		self.attention_type = attention_type
		self.output_size = output_size


		self.q_dense = tf.keras.layers.Dense(self.d_model, activation = self.feature_activation)
		self.k_dense = tf.keras.layers.Dense(self.d_model, activation = self.feature_activation)
		self.v_dense = tf.keras.layers.Dense(self.d_model, activation = self.feature_activation)
		self.out_dense = tf.keras.layers.Dense(self.output_size, activation = self.output_activation)
		if self.attention_type == "local_1d":
			self.query_block_length = query_block_length
			assert self.query_block_length is not None, "Did not provide query block length"

		if self.attention_type == "local_2d":
			self.query_kernel_size = query_kernel_size


		assert self.d_model % self.num_heads == 0, "D_model and Number of Heads do not match"

	def dot_product_attention(self, q, k, v, bias = None):
		attention = tf.einsum("...ik,...jk->...ij", q, k)
		if bias is not None:
			attention += bias
		softmax_attention_score = tf.math.softmax(attention)
		return tf.matmul(softmax_attention_score, v)

	def call(self, X):
		# Input Shape: [batch_size, height, width, channels]

		X_shape = tf.shape(X)

		b, l, c = X_shape[0], X_shape[1], X_shape[2]

		q_features = self.q_dense(X) # output: (batch_size, height * width * channels, d_model)
		k_features = self.k_dense(X) # output: (batch_size, height * width * channels, d_model)
		v_features = self.v_dense(X) # output: (batch_size, height * width * channels, d_model)

		q_features /= (self.d_model ** .5)

		q_heads = tf.reshape(q_features, (b, self.num_heads, -1, self.query_size))
		k_heads = tf.reshape(k_features, (b, self.num_heads, -1, self.query_size))
		v_heads = tf.reshape(v_features, (b, self.num_heads, -1, self.query_size))

		if self.attention_type == "global":

			attention = tf.einsum("...ik,...jk->...ij", q_heads, k_heads)

			softmax_attention_score = tf.math.softmax(attention)

			self_attentioned_value = tf.matmul(softmax_attention_score, v_heads)

		elif self.attention_type == "local_1d":

			if l % self.query_block_length != 0:

				padding = [
					[0, 0],
					[0, 0],
					[0, self.query_block_length - ((l) % self.query_block_length)],
					[0, 0]
				]

				padded_q_heads = tf.pad(q_heads, padding) # output: (batch_size, num_heads, (h*w*c) padded, query_size)
				padded_k_heads = tf.pad(k_heads, padding) # output: (batch_size, num_heads, (h*w*c) padded, query_size)
				padded_v_heads = tf.pad(v_heads, padding) # output: (batch_size, num_heads, (h*w*c) padded, query_size)
			
			else:

				padded_q_heads = q_heads
				padded_k_heads = k_heads
				padded_v_heads = v_heads


			# padding = np.zeros((len(np.shape(q_heads)), 2))

			# print(np.int32(self.query_block_length - ((h*w) % self.query_block_length)))

			# padding[-2, 1] = np.int32(self.query_block_length - ((h*w) % self.query_block_length))

			padded_q_heads = tf.reshape(padded_q_heads,
			(b,
			self.num_heads, 
			-1,
			self.query_size,
			self.query_block_length)) # output: (batch_size, num_heads, num_blocks, query_size, query_block_length)

			padded_k_heads = tf.reshape(padded_k_heads,
			(b,
			self.num_heads, 
			-1,
			self.query_size,
			self.query_block_length)) # output: (batch_size, num_heads, num_blocks, query_size, query_block_length)

			padded_v_heads = tf.reshape(padded_v_heads,
			(b,
			self.num_heads, 
			-1,
			self.query_size,
			self.query_block_length)) # output: (batch_size, num_heads, num_blocks, query_size, query_block_length)

			attention = tf.einsum("...ik,...jk->...ij", padded_q_heads, padded_k_heads)

			softmax_attention_score = tf.math.softmax(attention)

			self_attentioned_value_unshaped_padded = tf.matmul(softmax_attention_score, padded_v_heads)

			self_attentioned_value_padded = tf.reshape(self_attentioned_value_unshaped_padded, (b, self.num_heads, -1, self.query_size))

			self_attentioned_value = self_attentioned_value_padded[:, :, :(l), :]

		elif self.attention_type == "local_2d":
			# TODO: implement local 2d
			pass
			

		concatted_head_value = tf.reshape(self_attentioned_value, (b, -1, self.d_model))

		output = self.out_dense(concatted_head_value)

		return output

















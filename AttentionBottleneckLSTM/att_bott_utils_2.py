from imports import tf, Iterable, np, json, pd, date, nx, tqdm, NoDependency
from Conv2DmhaLSTMCell import Conv2DmhaLSTMCell
from MultiHeadGraphAttentionLSTMCell import MultiHeadGraphAttentionLSTMCell

def create_att_bottleneck_seq2seq(
	# global arguments
	encoer_layer_units: Iterable,
	sequence_length: int,
	num_heads: int,
	num_pad_tokens: int,
	join_layer: int,
	refresh_pad_tokens: bool,
	d_model: int,
	output_size: int,
	return_sequence_length: int,
	# image arguments
	image_dims: dict, # dictionary format to bypass dependency issues when saving model weights
	# https://github.com/tensorflow/tensorflow/issues/36916
	kernel_size: tuple,
	maxpool_kernel: int,
	# graph arguments
	input_shape_nodes: tuple,
	input_shape_edges: tuple,
	sequence_length_graph: int,
	residual: bool,
	use_bias: bool,
	# defaulted image arguments
	activation = tf.keras.activations.tanh,
	recurrent_activation = tf.keras.activations.hard_sigmoid,
	mha_feature_activation: str = "relu",
	mha_output_activation: str = "linear",
	use_maxpool: bool = True,
	# defaulted graph arguments
	concat_output: bool = False,
	# global defaulted arguments
	name: str = "Seq2SeqAttBottleneck"):

	assert join_layer < len(encoer_layer_units), "Layer to join is out of bounds"

	# image input
	input_layer_image = tf.keras.layers.Input(shape = [sequence_length,] + [image_dims["0"], image_dims["1"], 1])

	# graph inputs
	input_nodes = tf.keras.layers.Input(shape = input_shape_nodes) # nodes
	input_adj_mats = tf.keras.layers.Input(shape = input_shape_edges) # edges

	# init variables to keep track of image dimension and size
	curr_image_dims = image_dims
	curr_image_size = curr_image_dims["0"] * curr_image_dims["1"]

	# get batch shape for reshaping purposes
	b = tf.shape(input_layer_image)[0]

	# reassign inputs to new variables
	# to allow these tensors to be self-updated
	x_image = input_layer_image
	x_graph = input_nodes

	for i in range(len(encoer_layer_units) - 1):

		if i < (join_layer - 1):

			# Image
			mhaLSTM_cell_image = Conv2DmhaLSTMCell(
				units = encoer_layer_units[i],
				num_heads = num_heads,
				d_model = d_model,
				image_dims = NoDependency([curr_image_dims["0"], curr_image_dims["1"], 1]),
				kernel_size = kernel_size,
				name = f"{name}_mhaLSTMCell_{i}",
				activation = activation,
				recurrent_activation = recurrent_activation,
				mha_feature_activation = mha_feature_activation,
				mha_output_activation = mha_output_activation
			)
			x_image = tf.keras.layers.RNN(
				mhaLSTM_cell_image,
				return_sequences = True
			)(x_image)

			# update image dimensions if maxpool
			if use_maxpool:
				x_image = tf.keras.layers.MaxPool3D((1,maxpool_kernel,maxpool_kernel))(x_image)
				x_image = tf.keras.layers.LayerNormalization()(x_image)
				curr_image_dims["0"] = curr_image_dims["0"] // maxpool_kernel
				curr_image_dims["1"] = curr_image_dims["1"] // maxpool_kernel
				curr_image_size = curr_image_dims["0"] * curr_image_dims["1"]

			# Graph
			mhgaLSTM_cell_graph = MultiHeadGraphAttentionLSTMCell(
				units = encoer_layer_units[i],
				num_heads = num_heads,
				sequence_length = sequence_length_graph,
				output_size = d_model,
				residual = residual,
				concat_output = concat_output,
				use_bias = use_bias,
				name = f"{name}_cell_{i}"
			)
			x_graph = tf.keras.layers.RNN(
				mhgaLSTM_cell_graph,
				return_sequences = True
			)((x_graph, input_adj_mats))

			x_graph = tf.keras.layers.LayerNormalization()(x_graph)


		# reshape tokens and first pass through MultiHeadAttentionLSTMCell
		elif i == (join_layer - 1):

			# Image
			mhaLSTM_cell_image = Conv2DmhaLSTMCell(
				units = encoer_layer_units[i],
				num_heads = num_heads,
				d_model = d_model,
				image_dims = NoDependency([curr_image_dims["0"], curr_image_dims["1"], 1]),
				kernel_size = kernel_size,
				name = f"{name}_mhaLSTMCell_{i}",
				activation = activation,
				recurrent_activation = recurrent_activation,
				mha_feature_activation = mha_feature_activation,
				mha_output_activation = mha_output_activation
			)
			x_image = tf.keras.layers.RNN(
				mhaLSTM_cell_image,
				return_sequences = False
			)(x_image)

			# update image dimensions if maxpool
			if use_maxpool:
				x_image = tf.keras.layers.MaxPool2D((maxpool_kernel,maxpool_kernel))(x_image)
				x_image = tf.keras.layers.LayerNormalization()(x_image)
				curr_image_dims["0"] = curr_image_dims["0"] // maxpool_kernel
				curr_image_dims["1"] = curr_image_dims["1"] // maxpool_kernel
				curr_image_size = curr_image_dims["0"] * curr_image_dims["1"]

			# Graph
			mhgaLSTM_cell_graph = MultiHeadGraphAttentionLSTMCell(
				units = encoer_layer_units[i],
				num_heads = num_heads,
				sequence_length = sequence_length_graph,
				output_size = d_model,
				residual = residual,
				concat_output = concat_output,
				use_bias = use_bias,
				name = f"{name}_cell_{i}"
			)
			x_graph = tf.keras.layers.RNN(
				mhgaLSTM_cell_graph,
				return_sequences = False
			)((x_graph, input_adj_mats))

			x_graph = tf.keras.layers.LayerNormalization()(x_graph)

			# flatten image x and y dimensions
			#
			# curr_image_size must be provided
			# to ensure that the tensor shape for dimension 2 is not None.
			# If dimension 2 is None, concatenation along that dimension would propagate None shape
			x_image_tokens = tf.reshape(x_image, (b, curr_image_size, d_model))

			self_attention_tokens = tf.concat([x_image_tokens, tf.zeros((b, num_pad_tokens, d_model)), x_graph], axis = 1)

		# self attention of tokens through MultiHeadAttention
		else:

			self_attention_tokens = tf.keras.layers.MultiHeadAttention(
				num_heads = num_heads,
				key_dim = d_model,
				attention_axes = 1
			)(self_attention_tokens, self_attention_tokens)

			self_attention_tokens = tf.keras.layers.LayerNormalization()(self_attention_tokens)

			# if refresh_pad_tokens, reset pad tokens to zeros.
			if refresh_pad_tokens:
				self_attention_tokens = tf.concat(
					[
						self_attention_tokens[:, :curr_image_size, :],
						tf.zeros((b, num_pad_tokens, d_model)),
						self_attention_tokens[:, curr_image_size + num_pad_tokens:, :]],
						axis = 1)

	# Last MultiHeadAttention
	mhaLSTM_2 = tf.keras.layers.MultiHeadAttention(
		num_heads = num_heads,
		key_dim = d_model,
		attention_axes = 1
	)(self_attention_tokens, self_attention_tokens)

	mhaLSTM_2 = tf.keras.layers.LayerNormalization()(mhaLSTM_2)

	# separate tokens based on position
	image_tokens = mhaLSTM_2[:, :curr_image_size, :]

	graph_tokens = mhaLSTM_2[:, curr_image_size + num_pad_tokens:, :]

	# sum by d_model to preserve as much modality-specific info as possible
	image_tokens_sum = tf.reduce_sum(image_tokens, axis = 2)

	graph_tokens_sum = tf.reduce_sum(graph_tokens, axis = 2)

	# predict separate regression outputs for each modality and average results
	image_dense = tf.keras.layers.Dense(d_model, activation = "linear")(image_tokens_sum)

	graph_dense = tf.keras.layers.Dense(d_model, activation = "linear")(graph_tokens_sum)

	encoded_vector = (image_dense + graph_dense) / tf.constant(2.)

	# Generic Transformer from "Attention is All You Need": https://arxiv.org/pdf/1706.03762.pdf

	outputs = NoDependency([])

	x = tf.zeros_like(encoded_vector)

	for i in range(return_sequence_length):

		# TODO: Add Positional Encoding

		y = tf.keras.layers.MultiHeadAttention(
			num_heads = num_heads,
			key_dim = d_model
		)(x, x)

		x = tf.keras.layers.LayerNormalization()(x + y)

		y = tf.keras.layers.MultiHeadAttention(
			num_heads = num_heads,
			key_dim = d_model
		)(x, encoded_vector)

		x = tf.keras.layers.LayerNormalization()(x + y)

		y = tf.keras.layers.Dense(d_model, activation = "relu")(x)

		x = tf.keras.layers.LayerNormalization()(x + y)

		o = tf.keras.layers.Dense(output_size, activation = "linear")(x)

		outputs.append(o)

	return tf.keras.models.Model(inputs = [input_layer_image, input_nodes, input_adj_mats], outputs = outputs, name = name)



		


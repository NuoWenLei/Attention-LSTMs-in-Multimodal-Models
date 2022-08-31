from imports import tf, Iterable, np, json, pd, date, nx, tqdm, NoDependency, kt
from Conv2DmhaLSTMCell import Conv2DmhaLSTMCell
from MultiHeadGraphAttentionLSTMCell import MultiHeadGraphAttentionLSTMCell

def att_bottleneck_model_builder(hp):

	# global arguments
	layer_num_per_unit = hp.Choice("LAYER_UNITS", [16, 32])
	layer_units = (layer_num_per_unit, layer_num_per_unit, layer_num_per_unit)
	sequence_length = hp.Fixed("SEQ_LENGTH", 7)
	num_heads = hp.Fixed("NUM_HEADS", 8)
	num_pad_tokens = hp.Int("NUM_PAD_TOKENS", 8, 64, step = 8)
	join_layer = hp.Fixed("JOIN_LAYER", 2)
	refresh_pad_tokens = False
	d_model = hp.Choice("D_MODEL", [32, 64])
	output_size = hp.Fixed("OUTPUT_SIZE", 49)
	# image arguments
	image_dims = {
		"0": 64,
		"1": 64,
		"2": 1
	} # dictionary format to bypass dependency issues when saving model weights
	# https://github.com/tensorflow/tensorflow/issues/36916
	kernel_size = hp.Int("CONVLSTM_KERNEL_SIZE", 3, 7, step = 2)
	maxpool_kernel = hp.Fixed("MAXPOOL_KERNEL", 3)
	out_activation_image = hp.Choice("IMAGE_ACTIVATION_OUT", ["linear", "relu", "sigmoid", "tanh"])
	# graph arguments
	input_shape_nodes = (7, 49, 5)
	input_shape_edges = (7, 49, 49)
	sequence_length_graph = hp.Fixed("SEQ_LENGTH_GRAPH", 49)
	residual = True
	use_bias = True
	out_activation_graph = hp.Choice("GRAPH_ACTIVATION_OUT", ["linear", "relu", "sigmoid", "tanh"])
	# defaulted image arguments
	# activation = tf.keras.activations.tanh,
	# recurrent_activation = tf.keras.activations.hard_sigmoid,
	# mha_feature_activation: str = "relu",
	# mha_output_activation: str = "linear",
	use_maxpool = True
	use_layer_norm = hp.Choice("LAYER_NORM", ["Yes", "No"])
	attention_norm = hp.Choice("ATTENTION_LAYER_NORM", ["Yes", "No"])
	# defaulted graph arguments
	concat_output = False
	# global defaulted arguments
	name = "Conv2DAttentionLSTMModel"

	# compile arguments
	lr = hp.Choice("LEARNING_RATE", [0.0001, 0.001, 0.01])

	assert join_layer < len(layer_units), "Layer to join is out of bounds"

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

	for i in range(len(layer_units) - 1):

		if i < (join_layer - 1):

			# Image
			# mhaLSTM_cell_image = Conv2DmhaLSTMCell(
			# 	units = layer_units[i],
			# 	num_heads = num_heads,
			# 	d_model = d_model,
			# 	image_dims = NoDependency([curr_image_dims["0"], curr_image_dims["1"], 1]),
			# 	kernel_size = kernel_size,
			# 	name = f"{name}_mhaLSTMCell_{i}",
			# 	activation = activation,
			# 	recurrent_activation = recurrent_activation,
			# 	mha_feature_activation = mha_feature_activation,
			# 	mha_output_activation = mha_output_activation
			# )
			# x_image = tf.keras.layers.RNN(
			# 	mhaLSTM_cell_image,
			# 	return_sequences = True
			# )(x_image)

			x_image = tf.keras.layers.ConvLSTM2D(
				filters = layer_units[i],
				kernel_size = kernel_size,
				padding = "same",
				return_sequences = True)(x_image)

			# update image dimensions if maxpool
			if use_maxpool:
				x_image = tf.keras.layers.MaxPool3D((1,maxpool_kernel,maxpool_kernel))(x_image)
				with hp.conditional_scope("LAYER_NORM", ["Yes"]):
					if use_layer_norm == "Yes":
						x_image = tf.keras.layers.LayerNormalization()(x_image)
				curr_image_dims["0"] = curr_image_dims["0"] // maxpool_kernel
				curr_image_dims["1"] = curr_image_dims["1"] // maxpool_kernel
				curr_image_size = curr_image_dims["0"] * curr_image_dims["1"]

			# Graph
			mhgaLSTM_cell_graph = MultiHeadGraphAttentionLSTMCell(
				units = layer_units[i],
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

			with hp.conditional_scope("LAYER_NORM", ["Yes"]):
				if use_layer_norm == "Yes":
					x_graph = tf.keras.layers.LayerNormalization()(x_graph)

			# x_graph = tf.keras.layers.LayerNormalization()(x_graph)


		# reshape tokens and first pass through MultiHeadAttentionLSTMCell
		elif i == (join_layer - 1):

			# Image
			# mhaLSTM_cell_image = Conv2DmhaLSTMCell(
			# 	units = layer_units[i],
			# 	num_heads = num_heads,
			# 	d_model = d_model,
			# 	image_dims = NoDependency([curr_image_dims["0"], curr_image_dims["1"], 1]),
			# 	kernel_size = kernel_size,
			# 	name = f"{name}_mhaLSTMCell_{i}",
			# 	activation = activation,
			# 	recurrent_activation = recurrent_activation,
			# 	mha_feature_activation = mha_feature_activation,
			# 	mha_output_activation = mha_output_activation
			# )
			# x_image = tf.keras.layers.RNN(
			# 	mhaLSTM_cell_image,
			# 	return_sequences = False
			# )(x_image)

			x_image = tf.keras.layers.ConvLSTM2D(
				filters = d_model,
				kernel_size = kernel_size,
				padding = "same",
				return_sequences = False)(x_image)

			# update image dimensions if maxpool
			# if use_maxpool:
			# 	x_image = tf.keras.layers.MaxPool2D((maxpool_kernel,maxpool_kernel))(x_image)
			# 	with hp.conditional_scope("LAYER_NORM", ["Yes"]):
			# 		if use_layer_norm == "Yes":
			# 			x_image = tf.keras.layers.LayerNormalization()(x_image)
			# 	curr_image_dims["0"] = curr_image_dims["0"] // maxpool_kernel
			# 	curr_image_dims["1"] = curr_image_dims["1"] // maxpool_kernel
			# 	curr_image_size = curr_image_dims["0"] * curr_image_dims["1"]

			# Graph
			mhgaLSTM_cell_graph = MultiHeadGraphAttentionLSTMCell(
				units = d_model,
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

			with hp.conditional_scope("LAYER_NORM", ["Yes"]):
				if use_layer_norm == "Yes":
					x_graph = tf.keras.layers.LayerNormalization()(x_graph)

			# x_graph = tf.keras.layers.LayerNormalization()(x_graph)

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

			with hp.conditional_scope("ATTENTION_LAYER_NORM", ["Yes"]):
				if attention_norm == 'Yes':
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

	with hp.conditional_scope("ATTENTION_LAYER_NORM", ["Yes"]):
		if attention_norm == 'Yes':
			mhaLSTM_2 = tf.keras.layers.LayerNormalization()(mhaLSTM_2)

	# mhaLSTM_2 = tf.keras.layers.LayerNormalization()(mhaLSTM_2)

	# separate tokens based on position
	image_tokens = mhaLSTM_2[:, :curr_image_size, :]

	graph_tokens = mhaLSTM_2[:, curr_image_size + num_pad_tokens:, :]

	# sum by d_model to preserve as much modality-specific info as possible
	image_tokens_sum = tf.reduce_sum(image_tokens, axis = 2)

	graph_tokens_sum = tf.reduce_sum(graph_tokens, axis = 2)

	# predict separate regression outputs for each modality and average results
	image_dense = tf.keras.layers.Dense(output_size, activation = out_activation_image)(image_tokens_sum)

	graph_dense = tf.keras.layers.Dense(output_size, activation = out_activation_graph)(graph_tokens_sum)

	output = (image_dense + graph_dense) / tf.constant(2.)

	model = tf.keras.models.Model(inputs = [input_layer_image, input_nodes, input_adj_mats], outputs = output, name = name)

	opt = tf.keras.optimizers.Adam(learning_rate = lr)

	model.compile(
		loss = loss_func_3,
		optimizer = opt,
		metrics = ["mae", tf.keras.losses.cosine_similarity]
	)

	return model

def att_bottleneck_model_builder_with_json(hp):

	# global arguments
	layer_num_per_unit = hp["LAYER_UNITS"]
	layer_units = (layer_num_per_unit, layer_num_per_unit, layer_num_per_unit)
	sequence_length = hp["SEQ_LENGTH"]
	num_heads = hp["NUM_HEADS"]
	num_pad_tokens = hp["NUM_PAD_TOKENS"]
	join_layer = hp["JOIN_LAYER"]
	refresh_pad_tokens = False
	d_model = hp["D_MODEL"]
	output_size = hp["OUTPUT_SIZE"]
	# image arguments
	image_dims = {
		"0": 64,
		"1": 64,
		"2": 1
	} # dictionary format to bypass dependency issues when saving model weights
	# https://github.com/tensorflow/tensorflow/issues/36916
	kernel_size = hp["CONVLSTM_KERNEL_SIZE"]
	maxpool_kernel = hp["MAXPOOL_KERNEL"]
	out_activation_image = hp["IMAGE_ACTIVATION_OUT"]
	# graph arguments
	input_shape_nodes = (7, 49, 5)
	input_shape_edges = (7, 49, 49)
	sequence_length_graph = hp["SEQ_LENGTH_GRAPH"]
	residual = True
	use_bias = True
	out_activation_graph = hp["GRAPH_ACTIVATION_OUT"]
	# defaulted image arguments
	# activation = tf.keras.activations.tanh,
	# recurrent_activation = tf.keras.activations.hard_sigmoid,
	# mha_feature_activation: str = "relu",
	# mha_output_activation: str = "linear",
	use_maxpool = True
	use_layer_norm = hp["LAYER_NORM"]
	attention_norm = hp["ATTENTION_LAYER_NORM"]
	# defaulted graph arguments
	concat_output = False
	# global defaulted arguments
	name = "Conv2DAttentionLSTMModel"

	# compile arguments
	lr = hp["LEARNING_RATE"]

	assert join_layer < len(layer_units), "Layer to join is out of bounds"

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

	for i in range(len(layer_units) - 1):

		if i < (join_layer - 1):

			# Image
			# mhaLSTM_cell_image = Conv2DmhaLSTMCell(
			# 	units = layer_units[i],
			# 	num_heads = num_heads,
			# 	d_model = d_model,
			# 	image_dims = NoDependency([curr_image_dims["0"], curr_image_dims["1"], 1]),
			# 	kernel_size = kernel_size,
			# 	name = f"{name}_mhaLSTMCell_{i}",
			# 	activation = activation,
			# 	recurrent_activation = recurrent_activation,
			# 	mha_feature_activation = mha_feature_activation,
			# 	mha_output_activation = mha_output_activation
			# )
			# x_image = tf.keras.layers.RNN(
			# 	mhaLSTM_cell_image,
			# 	return_sequences = True
			# )(x_image)

			x_image = tf.keras.layers.ConvLSTM2D(
				filters = layer_units[i],
				kernel_size = kernel_size,
				padding = "same",
				return_sequences = True)(x_image)

			# update image dimensions if maxpool
			if use_maxpool:
				x_image = tf.keras.layers.MaxPool3D((1,maxpool_kernel,maxpool_kernel))(x_image)
				if use_layer_norm == "Yes":
					x_image = tf.keras.layers.LayerNormalization()(x_image)
				curr_image_dims["0"] = curr_image_dims["0"] // maxpool_kernel
				curr_image_dims["1"] = curr_image_dims["1"] // maxpool_kernel
				curr_image_size = curr_image_dims["0"] * curr_image_dims["1"]

			# Graph
			mhgaLSTM_cell_graph = MultiHeadGraphAttentionLSTMCell(
				units = layer_units[i],
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

			if use_layer_norm == "Yes":
				x_graph = tf.keras.layers.LayerNormalization()(x_graph)

			# x_graph = tf.keras.layers.LayerNormalization()(x_graph)


		# reshape tokens and first pass through MultiHeadAttentionLSTMCell
		elif i == (join_layer - 1):

			# Image
			# mhaLSTM_cell_image = Conv2DmhaLSTMCell(
			# 	units = layer_units[i],
			# 	num_heads = num_heads,
			# 	d_model = d_model,
			# 	image_dims = NoDependency([curr_image_dims["0"], curr_image_dims["1"], 1]),
			# 	kernel_size = kernel_size,
			# 	name = f"{name}_mhaLSTMCell_{i}",
			# 	activation = activation,
			# 	recurrent_activation = recurrent_activation,
			# 	mha_feature_activation = mha_feature_activation,
			# 	mha_output_activation = mha_output_activation
			# )
			# x_image = tf.keras.layers.RNN(
			# 	mhaLSTM_cell_image,
			# 	return_sequences = False
			# )(x_image)

			x_image = tf.keras.layers.ConvLSTM2D(
				filters = d_model,
				kernel_size = kernel_size,
				padding = "same",
				return_sequences = False)(x_image)

			# update image dimensions if maxpool
			# if use_maxpool:
			# 	x_image = tf.keras.layers.MaxPool2D((maxpool_kernel,maxpool_kernel))(x_image)
			# 	with hp.conditional_scope("LAYER_NORM", ["Yes"]):
			# 		if use_layer_norm == "Yes":
			# 			x_image = tf.keras.layers.LayerNormalization()(x_image)
			# 	curr_image_dims["0"] = curr_image_dims["0"] // maxpool_kernel
			# 	curr_image_dims["1"] = curr_image_dims["1"] // maxpool_kernel
			# 	curr_image_size = curr_image_dims["0"] * curr_image_dims["1"]

			# Graph
			mhgaLSTM_cell_graph = MultiHeadGraphAttentionLSTMCell(
				units = d_model,
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

			if use_layer_norm == "Yes":
				x_graph = tf.keras.layers.LayerNormalization()(x_graph)

			# x_graph = tf.keras.layers.LayerNormalization()(x_graph)

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

			if attention_norm == 'Yes':
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

	if attention_norm == 'Yes':
		mhaLSTM_2 = tf.keras.layers.LayerNormalization()(mhaLSTM_2)

	# mhaLSTM_2 = tf.keras.layers.LayerNormalization()(mhaLSTM_2)

	# separate tokens based on position
	image_tokens = mhaLSTM_2[:, :curr_image_size, :]

	graph_tokens = mhaLSTM_2[:, curr_image_size + num_pad_tokens:, :]

	# sum by d_model to preserve as much modality-specific info as possible
	image_tokens_sum = tf.reduce_sum(image_tokens, axis = 2)

	graph_tokens_sum = tf.reduce_sum(graph_tokens, axis = 2)

	# predict separate regression outputs for each modality and average results
	image_dense = tf.keras.layers.Dense(output_size, activation = out_activation_image)(image_tokens_sum)

	graph_dense = tf.keras.layers.Dense(output_size, activation = out_activation_graph)(graph_tokens_sum)

	output = (image_dense + graph_dense) / tf.constant(2.)

	model = tf.keras.models.Model(inputs = [input_layer_image, input_nodes, input_adj_mats], outputs = output, name = name)

	opt = tf.keras.optimizers.Adam(learning_rate = lr)

	model.compile(
		loss = loss_func_3,
		optimizer = opt,
		metrics = ["mae", tf.keras.losses.cosine_similarity]
	)

	return model

@tf.function
def loss_func(y_true, y_pred):
	cosine_sim = tf.keras.losses.cosine_similarity(y_true, y_pred)
	squared_diff_of_mean = tf.square(tf.reduce_mean(y_true, axis = -1) - tf.reduce_mean(y_pred, axis = -1))
	return cosine_sim + squared_diff_of_mean

@tf.function
def loss_func_2(y_true, y_pred):
	mse = tf.reduce_mean(tf.square(y_true - y_pred), axis = -1)
	squared_diff_of_mean = tf.square(tf.reduce_mean(y_true, axis = -1) - tf.reduce_mean(y_pred, axis = -1))
	return mse + squared_diff_of_mean

@tf.function
def loss_func_3(y_true, y_pred):
	cosine_sim = tf.keras.losses.cosine_similarity(y_true, y_pred)
	mse = tf.reduce_mean(tf.square(y_true - y_pred), axis = -1) * 100
	squared_diff_of_mean = tf.square(tf.reduce_mean(y_true, axis = -1) - tf.reduce_mean(y_pred, axis = -1)) * 50
	return mse + squared_diff_of_mean + cosine_sim

def load_sequential_data_image(maps_path: str,
metadata_path: str,
dataset_path: str,
image_x: int = 128,
image_y: int = 128,
num_days_per_sample: int = 7,
# filter_dates is used to make sure the sample size for image and graph are equal
filter_dates = None):

	# load raw data
	with open(maps_path, "rb") as f:
		maps = np.load(f)

	with open(metadata_path, "r") as meta_json:
		metadata = json.load(meta_json)

	# resize maps and pad to desired x and y
	maps = tf.image.resize_with_pad(maps, image_x, image_y).numpy()

	# read covid dataset
	df = pd.read_csv(dataset_path)

	# create dates formatted both m/d/y and Y/M/D to fit formats from maps and COVID dataset
	dates = [date(int("20" + str(y)), m, d).strftime("%-m/%-d/%y") for y, m, d in metadata]
	dates_ordered = [date(int("20" + str(y)), m, d).strftime("%Y/%m/%d") for y, m, d in metadata]

	# create dictionary that maps each date to an index
	image_idx_dictionary = dict([(d, i) for i, d in enumerate(dates)])

	print("Loading Image Indices...")

	# Get the desired order of images from COVID dataset date order
	image_indices = []
	for i, row in tqdm(df.iterrows()):
		image_indices.append(image_idx_dictionary[row["date"]])

	# link each row of COVID daataset to an image
	df["image_index"] = image_indices

	# create a dataframe that matches the corresponding m/d/y dates, Y/M/D dates, and image indices
	date_df = pd.DataFrame({"date": dates, "date_actual": dates_ordered})
	date_df["image_index"] = date_df.index

	# sort the dates dataframe by time from past to future
	sorted_date_df = date_df.sort_values("date_actual", ascending = True)


	# filter by filter_dates
	if filter_dates is not None:

		sorted_date_df = sorted_date_df[sorted_date_df["date_actual"].isin(filter_dates)]

	# for every day, flatten infection and death rates for the 49 state and store in lists
	raw_y_list_death = []
	raw_y_list_infection = []
	for d in sorted_date_df["date"].values:
		raw_y_list_death.append(df[df["date"] == d]["death_rate_from_population"].values)
		raw_y_list_infection.append(df[df["date"] == d]["infection_rate"].values)

	# sort maps by chronological order of dates
	raw_X = maps[sorted_date_df["image_index"]]

	raw_metadata = sorted_date_df
	raw_y_death = np.array(raw_y_list_death)
	raw_y_infection = np.array(raw_y_list_infection)

	# add sequence dimension to data
	# format each sample to have length num_days_per_sample for sequence dimension
	formatted_X_list = []
	formatted_y_list_death = []
	formatted_y_list_infection = []
	for i in range(raw_metadata.shape[0] - num_days_per_sample):
		formatted_X_list.append([n for n in range(i, i + num_days_per_sample)])

		formatted_y_list_death.append(raw_y_death[i + num_days_per_sample, ...])
		formatted_y_list_infection.append(raw_y_infection[i + num_days_per_sample, ...])

	formatted_X = np.array(formatted_X_list)
	formatted_y_death = np.array(formatted_y_list_death)
	formatted_y_infection = np.array(formatted_y_list_infection)

	# return results
	return formatted_X, formatted_y_death, formatted_y_infection, raw_X

def load_graph_data(covid_data_path, flight_data_path):
	# load raw data
	flight_df = pd.read_csv(flight_data_path)
	covid_df = pd.read_csv(covid_data_path)
	
	# reformat dates
	covid_df["adjusted_date"] = [date(int("20" + y), int(m), int(d)).strftime("%Y/%m/%d") for m, d, y in covid_df["date"].str.split("/")]

	# find dates missing from flight data
	# filter covid_df to only include dates with flight data
	adj_dates = set(flight_df.columns[2:])
	covid_dates = set(covid_df["adjusted_date"].values)
	adj_dates_lacked = covid_dates.difference(adj_dates)
	covid_df = covid_df[~covid_df["adjusted_date"].isin(list(adj_dates_lacked))]

	# create adjacency matrices by creating networkX Graphs
	adj_matrices = []
	for d in flight_df.columns[2:]:
		G = nx.from_pandas_edgelist(df = flight_df, source = "state_from", target = "state_to", edge_attr = d)
		A = nx.adjacency_matrix(G, weight = d)
		adj_matrices.append(A.todense())
	ADJ_MATRICES = np.array(adj_matrices)

	# return results
	# ADJ_MATRICES are like edges
	# covid_df are like nodes
	return ADJ_MATRICES, covid_df

def load_sequential_data_graph(covid_data_path, flight_data_path, num_days_per_sample = 7):
	# Load non-sequential graph data
	ADJ_MATRICES, covid_df = load_graph_data(covid_data_path, flight_data_path)

	# create unique dates
	sorted_unique_dates = np.sort(covid_df["adjusted_date"].unique())

	# init sequence lists
	print("Generating Sequential Data...")
	formatted_X_list = []
	formatted_adj_mat_list = []
	formatted_y_list_infection = []
	formatted_y_list_death = []

	# define valid cols
	valid_cols = ["Population", "confirm_value", "death_value", "infection_rate", "death_rate_from_population"]

	print("Loading Sequential Graph Data...")

	# add sequence dimension to graph data
	for i in tqdm(range(sorted_unique_dates.shape[0] - num_days_per_sample)):
		formatted_X_list.append([covid_df[covid_df["adjusted_date"] == d][valid_cols].values for d in sorted_unique_dates[i:i + num_days_per_sample]])

		formatted_adj_mat_list.append(ADJ_MATRICES[i:i + num_days_per_sample])

		formatted_y_list_infection.append(covid_df[covid_df["adjusted_date"] == sorted_unique_dates[i + num_days_per_sample]]["infection_rate"])

		formatted_y_list_death.append(covid_df[covid_df["adjusted_date"] == sorted_unique_dates[i + num_days_per_sample]]["death_rate_from_population"])

	formatted_X = np.array(formatted_X_list)
	formatted_adj_mat = np.array(formatted_adj_mat_list)
	formatted_y_infection = np.array(formatted_y_list_infection)
	formatted_y_death = np.array(formatted_y_list_death)

	# return data
	return (formatted_X, formatted_adj_mat, formatted_y_infection, formatted_y_death), sorted_unique_dates

def load_sequential_data(
	maps_path: str,
	metadata_path: str,
	covid_data_path: str,
	flight_data_path: str,
	image_x: int = 128,
	image_y: int = 128,
	num_days_per_sample = 7,
	return_dates = False):

	# create graph sequence data
	graph_data, unique_dates = load_sequential_data_graph(
		covid_data_path,
		flight_data_path,
		num_days_per_sample
	)

	# create image sequence data filtered by graph sequence unique dates
	image_data = load_sequential_data_image(
		maps_path,
		metadata_path,
		covid_data_path,
		image_x,
		image_y,
		num_days_per_sample,
		filter_dates = unique_dates
	)
	if return_dates:
		return image_data, graph_data, unique_dates
	return image_data, graph_data

def build_tuner(name, dir, max_epochs):
	tuner = kt.Hyperband(
		att_bottleneck_model_builder,
		objective = kt.Objective("val_loss", direction = "min"),
		max_epochs = max_epochs,
		directory = dir,
		project_name = name
	)

	print(tuner.search_space_summary())

	return tuner
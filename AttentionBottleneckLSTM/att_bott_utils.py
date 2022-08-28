from imports import tf, Iterable, np, json, pd, date, nx, tqdm, NoDependency
from Conv2DmhaLSTMCell import Conv2DmhaLSTMCell
from MultiHeadGraphAttentionLSTMCell import MultiHeadGraphAttentionLSTMCell
# from MultiHeadAttentionLSTMCell import MultiHeadAttentionLSTMCell

def create_att_bottleneck_model(
	# global arguments
	layer_units: Iterable,
	sequence_length: int,
	num_heads: int,
	num_pad_tokens: int,
	join_layer: int,
	refresh_pad_tokens: bool,
	d_model: int,
	output_size: int,
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
	name: str = "Conv2DAttentionLSTMModel"):

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
			mhaLSTM_cell_image = Conv2DmhaLSTMCell(
				units = layer_units[i],
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

			x_graph = tf.keras.layers.LayerNormalization()(x_graph)


		# reshape tokens and first pass through MultiHeadAttentionLSTMCell
		elif i == (join_layer - 1):

			# Image
			mhaLSTM_cell_image = Conv2DmhaLSTMCell(
				units = layer_units[i],
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
	image_dense = tf.keras.layers.Dense(output_size, activation = "linear")(image_tokens_sum)

	graph_dense = tf.keras.layers.Dense(output_size, activation = "linear")(graph_tokens_sum)

	output = (image_dense + graph_dense) / tf.constant(2.)

	return tf.keras.models.Model(inputs = [input_layer_image, input_nodes, input_adj_mats], outputs = output, name = name)


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
	num_days_per_sample = 7):

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
	
	return image_data, graph_data


def create_flow(
	# global
	batch_size,
	# images
	X_indices,
	raw_X,
	# maps
	X_nodes,
	X_edges,
	# target
	y):
	index = 0
	while True:
		X_image_sample = []
		X_node_sample = []
		X_edge_sample = []
		y_sample = []

		# create batch sample per call
		for _ in range(batch_size):
			X_image_sample.append(raw_X[X_indices[index, ...]])
			X_node_sample.append(X_nodes[index, ...])
			X_edge_sample.append(X_edges[index, ...])
			y_sample.append(y[index, ...])
			index += 1

			# reset index when samples ran out
			if index >= X_indices.shape[0]:
				index = 0
				# shuffle samples after every index reset
				p = np.random.permutation(y.shape[0])
				X_indices = X_indices[p]
				X_nodes = X_nodes[p]
				X_edges = X_edges[p]
				y = y[p]

		# yield batch samples
		yield ((
			np.float32(X_image_sample),
			np.float32(X_node_sample),
			np.float32(X_edge_sample)
			),
			np.float32(y_sample))

# DEPRICATED FUNCTIONS

# def calc_kernel_size(image_dims, blocks_y, blocks_x):
# 	y_complete = (image_dims[0] % blocks_y == 0)
# 	x_complete = (image_dims[1] % blocks_x == 0)
# 	kernel_size = [image_dims[0] // blocks_y, image_dims[1] // blocks_x]

# 	if not y_complete:
# 		kernel_size[0] += 1

# 	if not x_complete:
# 		kernel_size[1] += 1

# 	return kernel_size

# DEPRICATED MODEL STRUCTURES

# def create_att_bottleneck_model(
# 	# global arguments
# 	layer_units: Iterable,
# 	sequence_length: int,
# 	num_heads: int,
# 	num_pad_tokens: int,
# 	join_layer: int,
# 	refresh_pad_tokens: bool,
# 	d_model: int,
# 	output_size: int,
# 	# image arguments
# 	image_dims: tuple,
# 	kernel_size: tuple,
# 	maxpool_kernel: int,
# 	# graph arguments
# 	input_shape_nodes: tuple,
# 	input_shape_edges: tuple,
# 	sequence_length_graph: int,
# 	residual: bool,
# 	use_bias: bool,
# 	# defaulted image arguments
# 	activation = tf.keras.activations.tanh,
# 	recurrent_activation = tf.keras.activations.hard_sigmoid,
# 	mha_feature_activation: str = "relu",
# 	mha_output_activation: str = "linear",
# 	use_maxpool: bool = True,
# 	# defaulted graph arguments
# 	concat_output: bool = False,
# 	# global defaulted arguments
# 	name: str = "Conv2DAttentionLSTMModel"):

# 	assert join_layer < len(layer_units), "Layer to join is out of bounds"

# 	# kernel_size = calc_kernel_size(image_dims, blocks_y, blocks_x)

# 	input_layer_image = tf.keras.layers.Input(shape = (sequence_length,) + image_dims)

# 	input_nodes = tf.keras.layers.Input(shape = input_shape_nodes)
# 	input_adj_mats = tf.keras.layers.Input(shape = input_shape_edges)

# 	curr_image_dims = list(image_dims)
# 	curr_image_size = curr_image_dims[0] * curr_image_dims[1]
# 	b = tf.shape(input_layer_image)[0]

# 	x_image = input_layer_image
# 	x_graph = input_nodes

# 	for i in range(len(layer_units) - 1):

# 		if i < join_layer:

# 			# Image
# 			mhaLSTM_cell_image = Conv2DmhaLSTMCell(
# 				units = layer_units[i],
# 				num_heads = num_heads,
# 				d_model = d_model,
# 				image_dims = curr_image_dims,
# 				kernel_size = kernel_size,
# 				name = f"{name}_mhaLSTMCell_{i}",
# 				activation = activation,
# 				recurrent_activation = recurrent_activation,
# 				mha_feature_activation = mha_feature_activation,
# 				mha_output_activation = mha_output_activation
# 			)
# 			x_image = tf.keras.layers.RNN(
# 				mhaLSTM_cell_image,
# 				return_sequences = True
# 			)(x_image)

# 			if use_maxpool:
# 				x_image = tf.keras.layers.MaxPool3D((1,maxpool_kernel,maxpool_kernel))(x_image)
# 				x_image = tf.keras.layers.LeakyReLU()(x_image)
# 				x_image = tf.keras.layers.LayerNormalization()(x_image)
# 				curr_image_dims[0] = curr_image_dims[0] // maxpool_kernel
# 				curr_image_dims[1] = curr_image_dims[1] // maxpool_kernel
# 				curr_image_size = curr_image_dims[0] * curr_image_dims[1]

# 			# Graph
# 			mhgaLSTM_cell_graph = MultiHeadGraphAttentionLSTMCell(
# 				units = layer_units[i],
# 				num_heads = num_heads,
# 				sequence_length = sequence_length_graph,
# 				output_size = d_model,
# 				residual = residual,
# 				concat_output = concat_output,
# 				use_bias = use_bias,
# 				name = f"{name}_cell_{i}"
# 			)
# 			x_graph = tf.keras.layers.RNN(
# 				mhgaLSTM_cell_graph,
# 				return_sequences = True
# 			)((x_graph, input_adj_mats))

# 			x_graph = tf.keras.layers.LeakyReLU()(x_graph)
# 			x_graph = tf.keras.layers.LayerNormalization()(x_graph)


# 		# reshape tokens and first pass through MultiHeadAttentionLSTMCell
# 		elif i == join_layer:

# 			x_image_tokens = tf.reshape(x_image, (b, sequence_length, curr_image_size, d_model))

# 			full_token_sequence = tf.concat([x_image_tokens, tf.zeros((b, sequence_length, num_pad_tokens, d_model)), x_graph], axis = 2)

# 			mha_LSTMCell = MultiHeadAttentionLSTMCell(
# 				units = layer_units[i],
# 				num_heads = num_heads,
# 				d_model = d_model,
# 				num_tokens = curr_image_size + num_pad_tokens + sequence_length_graph,
# 				name = f"{name}_mhacell_{i}",
# 				activation = activation,
# 				recurrent_activation = recurrent_activation
# 			)

# 			self_attention_tokens = tf.keras.layers.RNN(
# 				mha_LSTMCell,
# 				return_sequences = True
# 			)(full_token_sequence)

# 			self_attention_tokens = tf.keras.layers.LeakyReLU()(self_attention_tokens)
# 			self_attention_tokens = tf.keras.layers.LayerNormalization()(self_attention_tokens)
			
# 			# implement refresh_pad_tokens argument that refreshes tokens between layers
# 			if refresh_pad_tokens:
# 				self_attention_tokens = tf.concat(
# 					[
# 						self_attention_tokens[:, :, :curr_image_size, :],
# 						tf.zeros((b, sequence_length, num_pad_tokens, d_model)),
# 						self_attention_tokens[:, :, curr_image_size + num_pad_tokens:, :]],
# 						axis = 2)

# 		# normal pass through with MultiHeadAttentionLSTMCell
# 		else:

# 			mha_LSTMCell = MultiHeadAttentionLSTMCell(
# 				units = layer_units[i],
# 				num_heads = num_heads,
# 				d_model = d_model,
# 				num_tokens = curr_image_size + num_pad_tokens + sequence_length_graph,
# 				name = f"{name}_mhacell_{i}",
# 				activation = activation,
# 				recurrent_activation = recurrent_activation
# 			)

# 			self_attention_tokens = tf.keras.layers.RNN(
# 				mha_LSTMCell,
# 				return_sequences = True
# 			)(self_attention_tokens)

# 			self_attention_tokens = tf.keras.layers.LeakyReLU()(self_attention_tokens)
# 			self_attention_tokens = tf.keras.layers.LayerNormalization()(self_attention_tokens)

# 			if refresh_pad_tokens:
# 				self_attention_tokens = tf.concat(
# 					[
# 						self_attention_tokens[:, :, :curr_image_size, :],
# 						tf.zeros((b, sequence_length, num_pad_tokens, d_model)),
# 						self_attention_tokens[:, :, curr_image_size + num_pad_tokens:, :]],
# 						axis = 2)

# 	mha_LSTMCell_out = MultiHeadAttentionLSTMCell(
# 		units = layer_units[-1],
# 		num_heads = num_heads,
# 		d_model = d_model,
# 		num_tokens = curr_image_size + num_pad_tokens + sequence_length_graph,
# 		name = f"{name}_mhacell_out",
# 		activation = activation,
# 		recurrent_activation = recurrent_activation
# 	)

# 	mhaLSTM_2 = tf.keras.layers.RNN(
# 		mha_LSTMCell_out,
# 		return_sequences = False
# 	)(self_attention_tokens)

# 	mhaLSTM_2 = tf.keras.layers.LeakyReLU()(mhaLSTM_2)
# 	mhaLSTM_2 = tf.keras.layers.LayerNormalization()(mhaLSTM_2)

# 	image_tokens = mhaLSTM_2[:, :curr_image_size, :]

# 	graph_tokens = mhaLSTM_2[:, curr_image_size + num_pad_tokens:, :]

# 	image_tokens_sum = tf.reduce_sum(image_tokens, axis = 1)

# 	graph_tokens_sum = tf.reduce_sum(graph_tokens, axis = 1)

# 	image_tokens_norm = tf.keras.layers.LayerNormalization()(image_tokens_sum)

# 	graph_tokens_norm = tf.keras.layers.LayerNormalization()(graph_tokens_sum)

# 	image_dense = tf.keras.layers.Dense(output_size, activation = "linear")(image_tokens_norm)

# 	graph_dense = tf.keras.layers.Dense(output_size, activation = "linear")(graph_tokens_norm)

# 	output = (image_dense + graph_dense) / tf.constant(2.)

# 	return tf.keras.models.Model(inputs = [input_layer_image, input_nodes, input_adj_mats], outputs = output, name = name)


# def create_conv_att_bottleneck_model(
# 	# global arguments
# 	layer_units: Iterable,
# 	sequence_length: int,
# 	num_heads: int,
# 	num_pad_tokens: int,
# 	join_layer: int,
# 	refresh_pad_tokens: bool,
# 	d_model: int,
# 	output_size: int,
# 	# image arguments
# 	image_dims: tuple,
# 	kernel_size: tuple,
# 	maxpool_kernel: int,
# 	# graph arguments
# 	input_shape_nodes: tuple,
# 	input_shape_edges: tuple,
# 	sequence_length_graph: int,
# 	residual: bool,
# 	use_bias: bool,
# 	# defaulted image arguments
# 	activation = tf.keras.activations.tanh,
# 	recurrent_activation = tf.keras.activations.hard_sigmoid,
# 	use_maxpool: bool = True,
# 	# defaulted graph arguments
# 	concat_output: bool = False,
# 	# global defaulted arguments
# 	name: str = "Conv2DAttentionLSTMModel"):

# 	assert join_layer < len(layer_units), "Layer to join is out of bounds"

# 	# kernel_size = calc_kernel_size(image_dims, blocks_y, blocks_x)

# 	input_layer_image = tf.keras.layers.Input(shape = (sequence_length,) + image_dims)

# 	input_nodes = tf.keras.layers.Input(shape = input_shape_nodes)
# 	input_adj_mats = tf.keras.layers.Input(shape = input_shape_edges)

# 	curr_image_dims = list(image_dims)
# 	curr_image_size = curr_image_dims[0] * curr_image_dims[1]
# 	b = tf.shape(input_layer_image)[0]

# 	x_image = input_layer_image
# 	x_graph = input_nodes

# 	for i in range(len(layer_units) - 1):

# 		if i < join_layer:

# 			# Image
# 			x_image = tf.keras.layers.ConvLSTM2D(
# 				filters = d_model,
# 				kernel_size = kernel_size,
# 				padding = "same",
# 				return_sequences = True,
# 				name = f"{name}_convLSTM_{i}"
# 			)(x_image)

# 			if use_maxpool:
# 				x_image = tf.keras.layers.MaxPool3D((1,maxpool_kernel,maxpool_kernel))(x_image)
# 				x_image = tf.keras.layers.LeakyReLU()(x_image)
# 				curr_image_dims[0] = curr_image_dims[0] // maxpool_kernel
# 				curr_image_dims[1] = curr_image_dims[1] // maxpool_kernel
# 				curr_image_size = curr_image_dims[0] * curr_image_dims[1]

# 			# Graph
# 			mhgaLSTM_cell_graph = MultiHeadGraphAttentionLSTMCell(
# 				units = layer_units[i],
# 				num_heads = num_heads,
# 				sequence_length = sequence_length_graph,
# 				output_size = d_model,
# 				residual = residual,
# 				concat_output = concat_output,
# 				use_bias = use_bias,
# 				name = f"{name}_cell_{i}"
# 			)
# 			x_graph = tf.keras.layers.RNN(
# 				mhgaLSTM_cell_graph,
# 				return_sequences = True
# 			)((x_graph, input_adj_mats))

# 			x_graph = tf.keras.layers.LeakyReLU()(x_graph)


# 		# reshape tokens and first pass through MultiHeadAttentionLSTMCell
# 		elif i == join_layer:

# 			x_image_tokens = tf.reshape(x_image, (b, sequence_length, curr_image_size, d_model))

# 			full_token_sequence = tf.concat([x_image_tokens, tf.zeros((b, sequence_length, num_pad_tokens, d_model)), x_graph], axis = 2)

# 			mha_LSTMCell = MultiHeadAttentionLSTMCell(
# 				units = layer_units[i],
# 				num_heads = num_heads,
# 				d_model = d_model,
# 				num_tokens = curr_image_size + num_pad_tokens + sequence_length_graph,
# 				name = f"{name}_mhacell_{i}",
# 				activation = activation,
# 				recurrent_activation = recurrent_activation
# 			)

# 			self_attention_tokens = tf.keras.layers.RNN(
# 				mha_LSTMCell,
# 				return_sequences = True
# 			)(full_token_sequence)

# 			# implement refresh_pad_tokens argument that refreshes tokens between layers
# 			if refresh_pad_tokens:
# 				self_attention_tokens = tf.concat(
# 					[
# 						self_attention_tokens[:, :, :curr_image_size, :],
# 						tf.zeros((b, sequence_length, num_pad_tokens, d_model)),
# 						self_attention_tokens[:, :, curr_image_size + num_pad_tokens:, :]],
# 						axis = 2)

# 		# normal pass through with MultiHeadAttentionLSTMCell
# 		else:

# 			mha_LSTMCell = MultiHeadAttentionLSTMCell(
# 				units = layer_units[i],
# 				num_heads = num_heads,
# 				d_model = d_model,
# 				num_tokens = curr_image_size + num_pad_tokens + sequence_length_graph,
# 				name = f"{name}_mhacell_{i}",
# 				activation = activation,
# 				recurrent_activation = recurrent_activation
# 			)

# 			self_attention_tokens = tf.keras.layers.RNN(
# 				mha_LSTMCell,
# 				return_sequences = True
# 			)(self_attention_tokens)

# 			if refresh_pad_tokens:
# 				self_attention_tokens = tf.concat(
# 					[
# 						self_attention_tokens[:, :, :curr_image_size, :],
# 						tf.zeros((b, sequence_length, num_pad_tokens, d_model)),
# 						self_attention_tokens[:, :, curr_image_size + num_pad_tokens:, :]],
# 						axis = 2)

# 	mha_LSTMCell_out = MultiHeadAttentionLSTMCell(
# 		units = layer_units[-1],
# 		num_heads = num_heads,
# 		d_model = d_model,
# 		num_tokens = curr_image_size + num_pad_tokens + sequence_length_graph,
# 		name = f"{name}_mhacell_out",
# 		activation = activation,
# 		recurrent_activation = recurrent_activation
# 	)

# 	mhaLSTM_2 = tf.keras.layers.RNN(
# 		mha_LSTMCell_out,
# 		return_sequences = False
# 	)(self_attention_tokens)

# 	image_tokens = mhaLSTM_2[:, :curr_image_size, :]

# 	graph_tokens = mhaLSTM_2[:, curr_image_size + num_pad_tokens:, :]

# 	image_tokens_sum = tf.reduce_sum(image_tokens, axis = 1)

# 	graph_tokens_sum = tf.reduce_sum(graph_tokens, axis = 1)

# 	image_dense = tf.keras.layers.Dense(output_size, activation = "linear")(image_tokens_sum)

# 	graph_dense = tf.keras.layers.Dense(output_size, activation = "linear")(graph_tokens_sum)

# 	output = (image_dense + graph_dense) / tf.constant(2.)

# 	return tf.keras.models.Model(inputs = [input_layer_image, input_nodes, input_adj_mats], outputs = output, name = name)

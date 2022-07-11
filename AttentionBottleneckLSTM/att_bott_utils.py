from imports import tf, Iterable, np, json, pd, date, nx, tqdm
from Conv2DAttentionLSTM import Conv2DmhaLSTMCell
from MultiHeadGraphAttentionLSTMCell import MultiHeadGraphAttentionLSTMCell
from MultiHeadAttentionLSTMCell import MultiHeadAttentionLSTMCell

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
	image_dims: tuple,
	kernel_size: tuple,
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

	# kernel_size = calc_kernel_size(image_dims, blocks_y, blocks_x)

	input_layer_image = tf.keras.layers.Input(shape = (sequence_length,) + image_dims)

	input_nodes = tf.keras.layers.Input(shape = input_shape_nodes)
	input_adj_mats = tf.keras.layers.Input(shape = input_shape_edges)

	curr_image_dims = list(image_dims)
	b = tf.shape(x_image)[0]

	x_image = input_layer_image
	x_graph = input_nodes

	for i in range(len(layer_units) - 1):

		if i < join_layer:

			# Image
			mhaLSTM_cell_image = Conv2DmhaLSTMCell(
				units = layer_units[i],
				num_heads = num_heads,
				d_model = d_model,
				image_dims = curr_image_dims,
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

			if use_maxpool:
				x_image = tf.keras.layers.MaxPool3D((1,2,2))(x_image)
				curr_image_dims[0] = curr_image_dims[0] // 2
				curr_image_dims[1] = curr_image_dims[1] // 2

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


		# reshape tokens and first pass through MultiHeadAttentionLSTMCell
		elif i == join_layer:

			x_image_tokens = tf.reshape(x_image, (b, sequence_length, -1, d_model))

			full_token_sequence = tf.concat([x_image_tokens, tf.zeros((b, sequence_length, num_pad_tokens, d_model)), x_graph], axis = 2)

			mha_LSTMCell = MultiHeadAttentionLSTMCell(
				units = layer_units[i],
				num_heads = num_heads,
				d_model = d_model,
				num_tokens = (curr_image_dims[0] * curr_image_dims[1]) + num_pad_tokens + sequence_length_graph,
				name = f"{name}_mhacell_{i}",
				activation = activation,
				recurrent_activation = recurrent_activation
			)

			self_attention_tokens = tf.keras.layers.RNN(
				mha_LSTMCell,
				return_sequences = True
			)(full_token_sequence, full_token_sequence)

			# implement refresh_pad_tokens argument that refreshes tokens between layers
			if refresh_pad_tokens:
				self_attention_tokens = tf.concat(
					[
						self_attention_tokens[:, :, :d_model, :],
						tf.zeros((b, sequence_length, num_pad_tokens, d_model)),
						self_attention_tokens[:, :, d_model + num_pad_tokens:, :]],
						axis = 2)

		# normal pass through with MultiHeadAttentionLSTMCell
		else:

			mha_LSTMCell = MultiHeadAttentionLSTMCell(
				units = layer_units[i],
				num_heads = num_heads,
				d_model = d_model,
				num_tokens = (curr_image_dims[0] * curr_image_dims[1]) + num_pad_tokens + sequence_length_graph,
				name = f"{name}_mhacell_{i}",
				activation = activation,
				recurrent_activation = recurrent_activation
			)

			self_attention_tokens = tf.keras.layers.RNN(
				mha_LSTMCell,
				return_sequences = True
			)(self_attention_tokens, self_attention_tokens)

			if refresh_pad_tokens:
				self_attention_tokens = tf.concat(
					[
						self_attention_tokens[:, :, :d_model, :],
						tf.zeros((b, sequence_length, num_pad_tokens, d_model)),
						self_attention_tokens[:, :, d_model + num_pad_tokens:, :]],
						axis = 2)

	mha_LSTMCell_out = MultiHeadAttentionLSTMCell(
		units = layer_units[-1],
		num_heads = num_heads,
		d_model = 1,
		num_tokens = (curr_image_dims[0] * curr_image_dims[1]) + num_pad_tokens + sequence_length_graph,
		name = f"{name}_mhacell_out",
		activation = activation,
		recurrent_activation = recurrent_activation
	)

	mhaLSTM_2 = tf.keras.layers.RNN(
		mha_LSTMCell_out,
		return_sequences = False
	)(self_attention_tokens)

	output = tf.keras.layers.Dense(output_size, activation = "linear")(mhaLSTM_2)

	return tf.keras.models.Model(inputs = [input_layer_image, input_nodes, input_adj_mats], outputs = output, name = name)

def calc_kernel_size(image_dims, blocks_y, blocks_x):
	y_complete = (image_dims[0] % blocks_y == 0)
	x_complete = (image_dims[1] % blocks_x == 0)
	kernel_size = [image_dims[0] // blocks_y, image_dims[1] // blocks_x]

	if not y_complete:
		kernel_size[0] += 1

	if not x_complete:
		kernel_size[1] += 1

	return kernel_size
	
def load_sequential_data_image(maps_path: str,
metadata_path: str,
dataset_path: str,
image_x: int = 128,
image_y: int = 128,
num_days_per_sample: int = 7):
	with open(maps_path, "rb") as f:
		maps = np.load(f)

	with open(metadata_path, "r") as meta_json:
		metadata = json.load(meta_json)

	maps = tf.image.resize_with_pad(maps, image_x, image_y).numpy()

	df = pd.read_csv(dataset_path)

	dates = [date(int("20" + str(y)), m, d).strftime("%-m/%-d/%y") for y, m, d in metadata]
	dates_ordered = [date(int("20" + str(y)), m, d).strftime("%Y/%m/%d") for y, m, d in metadata]

	image_idx_dictionary = dict([(d, i) for i, d in enumerate(dates)])

	print("Loading Image Indices...")

	image_indices = []
	for i, row in tqdm(df.iterrows()):
		image_indices.append(image_idx_dictionary[row["date"]])

	df["image_index"] = image_indices

	date_df = pd.DataFrame({"date": dates, "date_actual": dates_ordered})

	date_df["image_index"] = date_df.index

	sorted_date_df = date_df.sort_values("date_actual", ascending = True)

	raw_y_list = []
	for d in sorted_date_df["date"].values:
		raw_y_list.append(df[df["date"] == d]["death_rate_from_population"].values)

	raw_X = maps[sorted_date_df["image_index"]]
	raw_metadata = sorted_date_df
	raw_y = np.array(raw_y_list)

	formatted_X_list = []
	formatted_y_list = []
	for i in range(raw_metadata.shape[0] - num_days_per_sample):
		formatted_X_list.append([n for n in range(i, i + num_days_per_sample)])


		formatted_y_list.append(raw_y[i + num_days_per_sample, ...])

	formatted_X = np.array(formatted_X_list)
	formatted_y = np.array(formatted_y_list)

	return formatted_X, formatted_y, raw_X

def load_graph_data(covid_data_path, flight_data_path):
	flight_df = pd.read_csv(flight_data_path)
	covid_df = pd.read_csv(covid_data_path)
	covid_df["adjusted_date"] = [date(int("20" + y), int(m), int(d)).strftime("%Y/%m/%d") for m, d, y in covid_df["date"].str.split("/")]
	adj_dates = set(flight_df.columns[2:])
	covid_dates = set(covid_df["adjusted_date"].values)
	adj_dates_lacked = covid_dates.difference(adj_dates)
	covid_df = covid_df[~covid_df["adjusted_date"].isin(list(adj_dates_lacked))]

	adj_matrices = []
	for d in flight_df.columns[2:]:
		G = nx.from_pandas_edgelist(df = flight_df, source = "state_from", target = "state_to", edge_attr = d)
		A = nx.adjacency_matrix(G, weight = d)
		adj_matrices.append(A.todense())
	ADJ_MATRICES = np.array(adj_matrices)

	return ADJ_MATRICES, covid_df

def load_sequential_data_graph(covid_data_path, flight_data_path, num_days_per_sample = 7):
	ADJ_MATRICES, covid_df = load_graph_data(covid_data_path, flight_data_path)
	sorted_unique_dates = np.sort(covid_df["adjusted_date"].unique())

	print("Generating Sequential Data...")
	formatted_X_list = []
	formatted_adj_mat_list = []
	formatted_y_list_infection = []
	formatted_y_list_death = []
	valid_cols = ["Population", "confirm_value", "death_value", "infection_rate", "death_rate_from_population"]

	print("Loading Sequential Graph Data...")

	for i in tqdm(range(sorted_unique_dates.shape[0] - num_days_per_sample)):
		formatted_X_list.append([covid_df[covid_df["adjusted_date"] == d][valid_cols].values for d in sorted_unique_dates[i:i + num_days_per_sample]])

		formatted_adj_mat_list.append(ADJ_MATRICES[i:i + num_days_per_sample])

		formatted_y_list_infection.append(covid_df[covid_df["adjusted_date"] == sorted_unique_dates[i + num_days_per_sample]]["infection_rate"])

		formatted_y_list_death.append(covid_df[covid_df["adjusted_date"] == sorted_unique_dates[i + num_days_per_sample]]["death_rate_from_population"])

	formatted_X = np.array(formatted_X_list)
	formatted_adj_mat = np.array(formatted_adj_mat_list)
	formatted_y_infection = np.array(formatted_y_list_infection)
	formatted_y_death = np.array(formatted_y_list_death)

	return (formatted_X, formatted_adj_mat, formatted_y_infection, formatted_y_death)

def load_sequential_data(
	maps_path: str,
	metadata_path: str,
	covid_data_path: str,
	flight_data_path,
	image_x: int = 128,
	image_y: int = 128,
	num_days_per_sample = 7):

	image_data = load_sequential_data_image(
		maps_path,
		metadata_path,
		covid_data_path,
		image_x,
		image_y,
		num_days_per_sample)
	
	graph_data = load_sequential_data_graph(
		covid_data_path,
		flight_data_path,
		num_days_per_sample
	)

	return image_data, graph_data


def create_flow(X_indices, y, batch_size, raw_X):
	index = 0
	while True:
		X_sample = []
		y_sample = []
		for _ in range(batch_size):
			X_sample.append(raw_X[X_indices[index, ...]])
			y_sample.append(y[index, ...])
			index += 1
			if index >= X_indices.shape[0]:
				index = 0
				p = np.random.permutation(y.shape[0])
				X_indices = X_indices[p]
				y = y[p]

		yield np.float32(X_sample), np.float32(y_sample)



	
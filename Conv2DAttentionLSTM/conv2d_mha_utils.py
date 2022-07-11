from imports import tf, Iterable, np, json, pd, date
from Conv2DmhaLSTMCell import Conv2DmhaLSTMCell

def create_conv_mha_lstm_model(layer_units: Iterable,
num_heads: int,
d_model: int,
output_size: int,
image_dims: tuple,
kernel_size: tuple,
sequence_length: int,
activation = tf.keras.activations.tanh,
recurrent_activation = tf.keras.activations.hard_sigmoid,
mha_feature_activation: str = "relu",
mha_output_activation: str = "linear",
use_maxpool: bool = True,
use_out_dense: bool = True,
name: str = "Conv2DAttentionLSTMModel"):

	# kernel_size = calc_kernel_size(image_dims, blocks_y, blocks_x)

	input_layer = tf.keras.layers.Input(shape = (sequence_length,) + image_dims)

	curr_image_dims = list(image_dims)

	x = input_layer

	input_shape = tf.shape(x)

	x = tf.reshape(x, (input_shape[0], sequence_length, -1, 1))

	for i in range(len(layer_units) - 1):
		mhaLSTM_cell = Conv2DmhaLSTMCell(
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
		x = tf.keras.layers.RNN(
			mhaLSTM_cell,
			return_sequences = True
		)(x)

		if use_maxpool:
			x = tf.keras.layers.MaxPool2D()(x)
			curr_image_dims[0] = curr_image_dims[0] // 2
			curr_image_dims[1] = curr_image_dims[1] // 2
	
	mhaLSTM_out_cell = Conv2DmhaLSTMCell(
			units = layer_units[-1],
			num_heads = num_heads,
			d_model = output_size if not use_out_dense else d_model,
			image_dims = curr_image_dims,
			kernel_size = kernel_size,
			name = f"{name}_mhaLSTMCell_out",
			activation = activation,
			recurrent_activation = recurrent_activation,
			mha_feature_activation = mha_feature_activation,
			mha_output_activation = mha_output_activation
		)

	mhaLSTM_2 = tf.keras.layers.RNN(
		mhaLSTM_out_cell,
		return_sequences = False
	)(x)

	if use_out_dense:
		mhaLSTM_2_flattened = tf.reduce_mean(mhaLSTM_2, axis = [1, 2])
		output = tf.keras.layers.Dense(output_size, activation = "linear")(mhaLSTM_2_flattened)
	else:
		output = mhaLSTM_2

	return tf.keras.models.Model(inputs = input_layer, outputs = output, name = name)

def calc_kernel_size(image_dims, blocks_y, blocks_x):
	y_complete = (image_dims[0] % blocks_y == 0)
	x_complete = (image_dims[1] % blocks_x == 0)
	kernel_size = [image_dims[0] // blocks_y, image_dims[1] // blocks_x]

	if not y_complete:
		kernel_size[0] += 1

	if not x_complete:
		kernel_size[1] += 1

	return kernel_size
	
def load_sequential_data(maps_path: str,
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

	image_indices = []
	for i, row in df.iterrows():
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



	
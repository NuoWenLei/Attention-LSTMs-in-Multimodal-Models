from imports import tf, Iterable, np, json, pd, date
from ConvMhaLSTMCell import ConvMhaLSTMCell

def create_conv_mha_lstm_model(layer_units: Iterable,
num_heads: int,
d_model: int,
layer_output_size: int,
output_size: int,
image_dims: tuple,
sequence_length: int,
activation = tf.keras.activations.tanh,
recurrent_activation = tf.keras.activations.hard_sigmoid,
attention_type: str = "local_1d",
mha_feature_activation: str = "relu",
mha_output_activation: str = "linear",
query_block_length = None,
name: str = "ConvAttentionLSTMModel"):
	input_layer = tf.keras.layers.Input(shape = (sequence_length,) + image_dims)

	x = input_layer

	for i in range(len(layer_units) - 1):
		mhaLSTM_cell = ConvMhaLSTMCell(
			units = layer_units[i],
			num_heads = num_heads,
			d_model = d_model,
			output_size = layer_output_size,
			image_dims = image_dims,
			name = f"{name}_mhaLSTMCell_{i}",
			attention_type = attention_type,
			activation = activation,
			recurrent_activation = recurrent_activation,
			mha_feature_activation = mha_feature_activation,
			mha_output_activation = mha_output_activation,
			query_block_length = query_block_length
		)
		x = tf.keras.layers.RNN(
			mhaLSTM_cell,
			return_sequences = True
		)(x)
	
	mhaLSTM_out_cell = ConvMhaLSTMCell(
			units = layer_units[-1],
			num_heads = num_heads,
			d_model = d_model,
			output_size = output_size,
			image_dims = image_dims,
			name = f"{name}_mhaLSTMCell_out",
			attention_type = attention_type,
			activation = activation,
			recurrent_activation = recurrent_activation,
			mha_feature_activation = mha_feature_activation,
			mha_output_activation = mha_output_activation,
			query_block_length = query_block_length
		)

	mhaLSTM_2 = tf.keras.layers.RNN(
		mhaLSTM_out_cell,
		return_sequences = False
	)(x)

	output = tf.reduce_sum(mhaLSTM_2, axis = -2)

	return tf.keras.models.Model(inputs = input_layer, outputs = output, name = name)
	
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



	
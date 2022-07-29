from imports import tf, Iterable, np, json, pd, date, tqdm
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

	# image input
	input_layer = tf.keras.layers.Input(shape = (sequence_length,) + image_dims)

	# keep track of image dims
	curr_image_dims = list(image_dims)

	x = input_layer

	for i in range(len(layer_units) - 1):

		# process with conv2DmhaLSTMs for each layer
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

		# reduce image dims if use maxpool
		if use_maxpool:
			x = tf.keras.layers.MaxPool3D((1,2,2))(x)
			curr_image_dims[0] = curr_image_dims[0] // 2
			curr_image_dims[1] = curr_image_dims[1] // 2
	
	# Final LSTM
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

	# if use_out_dense, use extra dense layer to format prediction to right output size
	if use_out_dense:
		mhaLSTM_2_flattened = tf.reduce_mean(mhaLSTM_2, axis = [1, 2])
		output = tf.keras.layers.Dense(output_size, activation = "linear")(mhaLSTM_2_flattened)
	else:
		output = mhaLSTM_2

	return tf.keras.models.Model(inputs = input_layer, outputs = output, name = name)
	
def load_sequential_data(maps_path: str,
metadata_path: str,
dataset_path: str,
image_x: int = 128,
image_y: int = 128,
num_days_per_sample: int = 7,
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

def create_flow(X_indices, y, batch_size, raw_X):
	index = 0
	while True:
		X_sample = []
		y_sample = []
		
		# add batch sample to list
		for _ in range(batch_size):
			X_sample.append(raw_X[X_indices[index, ...]])
			y_sample.append(y[index, ...])
			index += 1

			# reset index and shuffle samples if reached end of dataset
			if index >= X_indices.shape[0]:
				index = 0
				p = np.random.permutation(y.shape[0])
				X_indices = X_indices[p]
				y = y[p]

		# yield batch samples
		yield np.float32(X_sample), np.float32(y_sample)

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

	
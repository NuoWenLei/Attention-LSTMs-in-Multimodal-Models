from imports import tf, Iterable
from ConvLSTMCell import ConvLSTMCell

def create_conv_lstm_model(layer_units: Iterable,
input_shape: tuple,
filters: int,
output_size: int,
kernel_size: int,
name = "ConvLSTMModel"):
	input_layer = tf.keras.layers.Input(shape = input_shape)

	x = input_layer

	for i in range(len(layer_units) - 1):
		convLSTM_cell = ConvLSTMCell(
			units = layer_units[i],
			filters = filters,
			kernel_size = kernel_size,
			name = f"{name}_convLSTMCell_{i}"
		)

		x = tf.keras.layers.RNN(convLSTM_cell,
		return_sequences=True)(x)
	
	convLSTM_out_cell = ConvLSTMCell(
		units = layer_units[-1],
		filters = output_size,
		kernel_size = kernel_size,
		name = f"{name}_convLSTMCell_out"
	)

	out = tf.keras.layers.RNN(convLSTM_out_cell)(x)

	return tf.keras.models.Model(inputs = input_layer, outputs = out, name = name)

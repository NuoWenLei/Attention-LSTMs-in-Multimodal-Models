from imports import tf, Iterable
from MultiHeadAttentionLSTMCell import MultiHeadAttentionLSTMCell

def create_conv_mha_lstm_model(layer_units: Iterable,
num_heads: int,
input_shape: tuple,
sequence_length: int,
hidden_size: int,
residual: bool,
name: str = "GraphAttentionLSTMModel"):
	input_layer = tf.keras.layers.Input(shape = input_shape)

	layer_shape = tf.shape(input_layer)

	if len(layer_shape) == 4:
		c = 1
		b, s, h, w = layer_shape
	else:
		b, s, h, w, c = layer_shape
		
	reshaped_input = tf.reshape(input_layer, (b, s, h * w * c))
	flattened_input_shape = (h * w * c,)

	x = reshaped_input

	for i in range(len(layer_units) - 1):
		mhaLSTM_cell = MultiHeadAttentionLSTMCell(
			units = layer_units[i],
			num_heads = num_heads,
			sequence_length = sequence_length,
			output_size = hidden_size,
			input_shape = flattened_input_shape,
			residual = residual,
			name = f"{name}_mhaLSTMCell_{i}"
		)
		x = tf.keras.layers.RNN(
			mhaLSTM_cell,
			return_sequences = True
		)(x)
	
	mhaLSTM_out_cell = MultiHeadAttentionLSTMCell(
		units = layer_units[-1],
		num_heads = num_heads,
		sequence_length = sequence_length,
		output_size = 1,
		input_shape = flattened_input_shape,
		residual = residual,
		name = f"{name}_mhaLSTMCell_out"
	)

	mhaLSTM_2 = tf.keras.layers.RNN(
		mhaLSTM_out_cell,
		return_sequences = False
	)(x)

	return tf.keras.models.Model(inputs = input_layer, outputs = mhaLSTM_2, name = name)
	


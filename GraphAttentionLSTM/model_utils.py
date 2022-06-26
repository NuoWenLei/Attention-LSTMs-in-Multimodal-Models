from typing import Iterable
from MultiHeadGraphAttentionLSTMCell import MultiHeadGraphAttentionLSTMCell, tf

def create_graph_attention_lstm_model(layer_units: Iterable,
num_heads: int,
sequence_length: int,
output_size: int,
residual: bool,
use_bias: bool,
concat_output: bool = False,
name: str = "GraphAttentionLSTMModel"):

	input_nodes = tf.keras.layers.Input(shape = (7, 49, 5))
	input_adj_mats = tf.keras.layers.Input(shape = (7, 49, 49))

	x = input_nodes

	for i in range(len(layer_units) - 1):
		mhgaLSTM_cell = MultiHeadGraphAttentionLSTMCell(
			units = layer_units[i],
			num_heads = num_heads,
			sequence_length = sequence_length,
			output_size = output_size,
			residual = residual,
			concat_output = concat_output,
			use_bias = use_bias,
			name = f"{name}_cell_{i}"
		)
		x = tf.keras.layers.RNN(
			mhgaLSTM_cell,
			return_sequences = True
		)((x, input_adj_mats))

	mhgaLSTM_out_cell = MultiHeadGraphAttentionLSTMCell(
		units = layer_units[-1],
		num_heads = num_heads,
		sequence_length = sequence_length,
		output_size = 1,
		residual = True,
		concat_output = False,
		use_bias = True,
		name = f"{name}_cell_out"
	)

	mhgaLSTM_2 = tf.keras.layers.RNN(mhgaLSTM_out_cell)((x, input_adj_mats))

	return tf.keras.models.Model(inputs = [input_nodes, input_adj_mats], outputs = mhgaLSTM_2)


# See https://github.com/keras-team/keras/blob/v2.9.0/keras/layers/rnn/base_conv_lstm.py#L197
# to create MultiHeadGraphAttentionLSTMCell layer
#
# See https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/layers/rnn/base_conv_lstm.py#L277
# for changes needed in base function for recurrent compatibility

from MultiHeadGraphAttention import MultiHeadGraphAttention
from imports import tf

class MultiHeadGraphAttentionLSTMCell(tf.keras.layers.Layer):
  
  def __init__(self,
               units: int,
               num_heads: int, 
               sequence_length: int,
               output_size: int,
               residual: bool,
               concat_output: bool,
               use_bias: bool,
               name: str,
               activation = tf.keras.activations.tanh,
               recurrent_activation = tf.keras.activations.hard_sigmoid):
    super().__init__(name = name)
    self.units = units
    self.num_heads = num_heads
    self.seq_len = sequence_length
    self.output_size = output_size
    self.state_size = [tf.TensorShape([self.seq_len, self.output_size]), tf.TensorShape([self.seq_len, self.output_size])]
    self.activation = activation
    self.recurrent_activation = recurrent_activation
    self.residual = residual
    self.use_bias = use_bias
    self.concat_output = concat_output

    (self.input_graph_attention_i, self.input_graph_attention_f,
     self.input_graph_attention_c, self.input_graph_attention_o) = [MultiHeadGraphAttention(num_heads = self.num_heads,
                                                           sequence_length = self.seq_len,
                                                           output_size = self.output_size,
                                                           activation = None,
                                                           residual = self.residual,
                                                           concat_output = self.concat_output,
                                                           use_bias = self.use_bias,
                                                           name = f"{name}_input_mha_{i}"
                                                           ) for i in range(4)]
    (self.recurrent_graph_attention_i, self.recurrent_graph_attention_f,
     self.recurrent_graph_attention_c, self.recurrent_graph_attention_o) = [MultiHeadGraphAttention(num_heads = self.num_heads,
                                                           sequence_length = self.seq_len,
                                                           output_size = self.output_size,
                                                           activation = None,
                                                           residual = self.residual,
                                                           concat_output = self.concat_output,
                                                           use_bias = False, # False because hidden state does not include bias value
                                                           name = f"{name}_recurrent_mha_{i}"
                                                           ) for i in range(4)]

  def call(self, inputs, states):
    
    prev_h = states[0]
    prev_c = states[1] 

    # Processing inputs
    x_i = self.input_graph_attention_i(inputs)
    x_f = self.input_graph_attention_f(inputs)
    x_c = self.input_graph_attention_c(inputs)
    x_o = self.input_graph_attention_o(inputs)

    # Processing hidden state
    h_i = self.recurrent_graph_attention_i(prev_h)
    h_f = self.recurrent_graph_attention_f(prev_h)
    h_c = self.recurrent_graph_attention_c(prev_h)
    h_o = self.recurrent_graph_attention_o(prev_h)

    # LSTM calculations
    i = tf.cast(self.recurrent_activation(x_i + h_i), tf.float32)
    f = tf.cast(self.recurrent_activation(x_f + h_f), tf.float32)
    g = tf.cast(self.activation(x_c + h_c), tf.float32)
    o = tf.cast(self.recurrent_activation(x_o + h_o), tf.float32)
    c = f * prev_c + i * g
    h = o * self.activation(c)

    return h, [h, c]
from imports import tf

class GraphAttentionHead(tf.keras.layers.Layer):

  def __init__(self, sequence_length: int, output_size: int, activation, residual: bool, use_bias: bool, name: str):
    super().__init__(name = name)
    self.seq_len = sequence_length
    self.output_size = output_size
    self.residual = residual
    self.use_bias = use_bias
    self.activation = activation
    self.bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)

    self.conv_sequence_feature_generation = tf.keras.layers.Conv1D(self.output_size, 1)
    self.conv_self_attention_dim_1 = tf.keras.layers.Conv1D(1, 1)
    self.conv_self_attention_dim_2 = tf.keras.layers.Conv1D(1, 1)
    self.leaky_relu = tf.keras.layers.LeakyReLU()
    self.bias_value = tf.Variable(self.bias_initializer(shape = (self.seq_len, 1)))

    if self.residual:
      self.conv_residual = tf.keras.layers.Conv1D(self.output_size, 1)

  def call(self, inputs):
    if self.use_bias:
      sequence = inputs[0]
      bias_mat = inputs[1]
    else:
      sequence = inputs
    
    # create feature tokens
    seq_feature_tokens = self.conv_sequence_feature_generation(sequence) # output shape: (seq_len, output_size)

    # create attention dimensions
    seq_att_dim_1 = self.conv_self_attention_dim_1(seq_feature_tokens) # output shape: (seq_len, 1)
    seq_att_dim_2 = self.conv_self_attention_dim_2(seq_feature_tokens) # output shape: (seq_len, 1)

    # cross add attention dimensions to create attention score
    non_softmax_attention = seq_att_dim_1 + tf.transpose(seq_att_dim_2, [0, 2, 1]) # output shape: (seq_len, seq_len)

    # add adjacency matrix or graph edges
    if self.use_bias:
      non_softmax_attention = non_softmax_attention + bias_mat # output shape: (seq_len, seq_len)

    # softmax to scale attention score
    softmax_attention_score = tf.keras.activations.softmax(non_softmax_attention) # output shape: (seq_len, seq_len)

    # apply attention score onto features
    attention_applied_sequence_features = tf.matmul(softmax_attention_score, seq_feature_tokens) # output shape: (seq_len, output_size)
    attention_applied_sequence_features_w_bias = attention_applied_sequence_features + self.bias_value # output shape: (seq_len, output_size)

    # add residual from input
    if self.residual:
      attention_applied_sequence_features_w_bias = attention_applied_sequence_features_w_bias + self.conv_residual(sequence)
    
    # activate output
    if self.activation:
      return self.activation(attention_applied_sequence_features_w_bias)

    return attention_applied_sequence_features_w_bias


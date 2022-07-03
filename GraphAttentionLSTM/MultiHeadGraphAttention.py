from GraphAttentionHead import GraphAttentionHead
from imports import tf

class MultiHeadGraphAttention(tf.keras.layers.Layer):
  
  def __init__(self, num_heads: int, sequence_length: int, output_size: int, activation, residual: bool, concat_output: bool, use_bias: bool, name: str):
    # The MultiHeadAttention mechanism implemented in GAT is quite odd
    # as it does not have a query dimension to split the inputs by.
    #
    # This means that the full input is passed on into each attention head.
    #
    # I am unsure if this is intentional or a place to be optimized.
    super().__init__(name = name)
    if concat_output:
      assert output_size % num_heads == 0, f"Output size {output_size} unable to be split evenly among {num_heads} heads"
    self.concat_output = concat_output
    self.output_size_per_head = output_size // num_heads
    self.num_heads = num_heads
    self.output_size = output_size
    self.seq_len = sequence_length
    self.activation = activation
    self.residual = residual
    self.use_bias = use_bias
    self.heads = [GraphAttentionHead(self.seq_len,
                                     self.output_size,
                                     self.activation,
                                     self.residual,
                                     self.use_bias,
                                     f"{name}_head_{i}"
                                     ) for i in range(self.num_heads)]
    
  def call(self, inputs):
    attns = []

    for head in self.heads:
      head_output = head(inputs)
      attns.append(head_output)

    if self.concat_output:
      output = tf.concat(attns, axis = -1)
    else:
      output = tf.add_n(attns) / self.num_heads
      
    return output
      
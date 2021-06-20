"""Defines XLNet model in tf.keras.API."""
import numpy as np

import tensorflow as tf
import math
from commons.layers import Projection
from commons.layers import FeedForwardNetwork
from commons.beam_search import NEG_INF


SEG_ID_CLS = 2

def cache_memory(inputs, mems=None, mem_len=0, reuse_len=0):
  """
  """
  if reuse_len > 0:
    inputs = inputs[:, :reuse_len]
  if mems is None:
    new_mem = inputs[:, -mem_len:]
  else:
    new_mem = tf.concat([mems, inputs[:, :reuse_len]], 1)[:, -mem_len:]
  return tf.stop_gradient(new_mem)


def rel_shift(inputs):
  """"""
  shape = tf.shape(inputs)
  padded = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [1, 0]])
  reshaped = tf.reshape(padded, [shape[0], shape[1], shape[3] + 1, shape[2]])
  sliced = reshaped[:, :, 1:]
  outputs = tf.reshape(sliced, shape)
  return outputs


class Attention(tf.keras.layers.Layer):
  """Multi-headed attention layer used in XLNet model. Jointly processes the
  content stream and the query stream by making them attend to the context.

  Note that attention weights are computed by combining three sources of
  information -- content, position (as in TransformerXL), and segment (to
  model the multi-segment input to XLNet).
  """
  def __init__(self, hidden_size, num_heads, dropout_rate_attention):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      dropout_rate_attention: float scalar, dropout rate applied on the
        query-to-reference attention matrix.
    """
    super(Attention, self).__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._dropout_rate_attention = dropout_rate_attention
    self._size_per_head = hidden_size // num_heads

    self._dense_layer_query = Projection(
        num_heads, self._size_per_head, mode='split')
    self._dense_layer_key_content = Projection(
        num_heads, self._size_per_head, mode='split')
    self._dense_layer_value = Projection(
        num_heads, self._size_per_head, mode='split')
    self._dense_layer_key_position = Projection(
        num_heads, self._size_per_head, mode='split')
    self._dense_layer_output = Projection(
        num_heads, self._size_per_head, mode='merge')
    self._attention_dropout_layer = tf.keras.layers.Dropout(
        dropout_rate_attention)

  def call(self,
           content_inputs,
           query_inputs,
           position_encoding,
           segment_encoding,
           segment_matrix,
           target_mapping,
           content_mask,
           query_mask,
           content_bias,
           position_bias,
           segment_bias,
           mems=None,
           training=False):
    """Computes new representations of content and query stream.

    Args:
      content_inputs: float tensor of shape [batch_size, q_seq_len,
        hidden_size], input content stream.
      query_inputs: float tensor of shape [batch_size, num_targets,
        hidden_size], input query stream.
      position_encoding: float tensor of shape [batch_size, m_seq_len +
        2 * q_seq_len, hidden_size], encodings of the relative position 
        information.
      segment_encoding: float tensor of shape [2, num_heads, size_per_head],
        embedding vectors of the binary information that whether two positions
        are from the same segment or not.
      segment_matrix: float tensor of shape [batch_size, q_seq_len, m_seq_len +
        q_seq_len], indicator matrix specifying if two positions are from the
        same segment or not.
      target_mapping: float tensor of shape [batch_size, num_targets,
        q_seq_len], one-hot encodings of the indices of prediction targets.
      content_mask: float tensor of shape [batch_size, 1, q_seq_len, m_seq_len +
        q_seq_len], permutation mask for the content stream.
      query_mask: float tensor of shape [batch_size, 1, q_seq_len, m_seq_len +
        q_seq_len], permutation mask for the query stream.
      content_bias: float tensor of shape [num_heads, size_per_head], bias to be
        added to the query sequences.
      position_bias: float tensor of shape [num_heads, size_per_head], bias to
        be added to the query sequences.
      segment_bias: float tensor of shape [num_heads, size_per_head], bias to
        be added to the query sequences.
      mems: (Optional) float tensor of shape [batch_size, m_seq_len, hidden_size
        ], encodings of the memory sequences from the previous block.
      training: (Optional) bool scalar, True if in training mode.

    Returns:
      content_output: float tensor of shape [batch_size, q_seq_len,
        hidden_size], output content stream.
      query_output: float tensor of shape [batch_size, num_targets,
        hidden_size], output query stream.
    """
    contexts = content_inputs if mems is None else tf.concat(
        [mems, content_inputs], 1)

    # [batch_size, m_seq_len + q_seq_len, num_heads, size_per_head]
    key_content = self._dense_layer_key_content(contexts)
    # [batch_size, m_seq_len + q_seq_len, num_heads, size_per_head]
    value = self._dense_layer_value(contexts)
    # [batch_size, m_seq_len + 2 * seq_len, num_heads, size_per_head]
    key_position = self._dense_layer_key_position(position_encoding)

    kwargs = {'key_content': key_content,
              'value': value,
              'key_position': key_position,
              'segment_encoding': segment_encoding,
              'segment_matrix': segment_matrix,
              'content_bias': content_bias,
              'position_bias': position_bias,
              'segment_bias': segment_bias,
              'training': training}

    # [batch_size, q_seq_len, num_heads, size_per_head]
    query = self._dense_layer_query(content_inputs)
    # [batch_size, q_seq_len, num_heads, size_per_head]
    content_outputs = self._compute_attention(
        query, content_mask, **kwargs)
    # [batch_size, q_seq_len, hidden_size]
    content_outputs = self._dense_layer_output(content_outputs)


    # [batch_size, num_targets, num_heads, size_per_head]
    query = self._dense_layer_query(query_inputs)
    # [batch_size, q_seq_len, num_heads, size_per_head]
    query = tf.einsum('NPHS,NPQ->NQHS', query, target_mapping)
    # [batch_size, q_seq_len, num_heads, size_per_head]
    query_outputs = self._compute_attention(
        query, query_mask, **kwargs)
    # [batch_size, num_targets, num_heads, size_per_head]
    query_outputs = tf.einsum('NQHS,NPQ->NPHS', query_outputs, target_mapping)
    # [batch_size, num_targets, hidden_size]
    query_outputs = self._dense_layer_output(query_outputs)

    return content_outputs, query_outputs

  def _compute_attention(self,
                         query,
                         token_mask,
                         key_content,
                         value,
                         key_position,
                         segment_encoding,
                         segment_matrix,
                         content_bias,
                         position_bias,
                         segment_bias,
                         training=False):
    """Computes attetion weights.

    Args:
      query: float tensor of shape [batch_size, q_seq_len, num_heads,
        size_per_head], multi-headed query sequences.
      token_mask: float tensor of shape [batch_size, 1, q_seq_len, m_seq_len +
        q_seq_len], populated with either 0 (for tokens to keep) or 1 (for
        tokens to be masked).
      key_content: float tensor of shape [batch_size, m_seq_len + q_seq_len,
        num_heads, size_per_head], multi-headed content-based key sequences.
      value: float tensor of shape [batch_size, m_seq_len + q_seq_len, num_heads
        , size_per_head], multi-headed value sequences.
      key_position: float tensor of shape [batch_size, m_seq_len + 2 * q_seq_len
        , num_heads, size_per_head], multi-headed position-based key sequences.
      segment_encoding: float tensor of shape [2, num_heads, size_per_head],
        embedding vectors of the binary information that whether two positions
        are from the same segment or not.
      segment_matrix: float tensor of shape [batch_size, q_seq_len, m_seq_len +
        q_seq_len], indicator matrix specifying if two positions are from the
        same segment or not.
      content_bias: float tensor of shape [num_heads, size_per_head], bias to be
        added to the query sequences.
      position_bias: float tensor of shape [num_heads, size_per_head], bias to
        be added to the query sequences.
      segment_bias: float tensor of shape [num_heads, size_per_head], bias to
        be added to the query sequences.
      training: (Optional) bool scalar, True if in training mode.

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, num_heads,
        size_per_head, the new representation of `query`.
    """
    # [batch_size, num_heads, q_seq_len, m_seq_len + q_seq_len]
    content_attention = tf.einsum('NQHS,NRHS->NHQR',
                                  query + content_bias,
                                  key_content)

    # [batch_size, num_heads, q_seq_len, m_seq_len + 2 * q_seq_len]
    position_attention = tf.einsum('NQHS,NRHS->NHQR',
                                   query + position_bias,
                                   key_position)

    # [batch_size, num_heads, q_seq_len, m_seq_len + q_seq_len]
    position_attention = rel_shift(position_attention)[
        ..., 1:1 + tf.shape(content_attention)[3]]
    
    # [batch_size, num_heads, q_seq_len, 2]
    segment_attention = tf.einsum('NQHS,GHS->NHQG',
                                  query + segment_bias,
                                  segment_encoding)

    attention_shape = tf.shape(content_attention)

    # [batch_size, num_heads, q_seq_len, m_seq_len + q_seq_len]
    segment_attention = tf.where(
        tf.broadcast_to(segment_matrix[:, tf.newaxis], attention_shape),
        tf.broadcast_to(segment_attention[..., 1:], attention_shape),
        tf.broadcast_to(segment_attention[..., :1], attention_shape))

    attention_weights = tf.multiply(
        content_attention + position_attention + segment_attention,
        1.0 / math.sqrt(float(self._size_per_head)))

    attention_weights += token_mask * NEG_INF
    attention_weights = tf.nn.softmax(attention_weights, axis=3)
    attention_weights = self._attention_dropout_layer(
        attention_weights, training=training)

    outputs = tf.einsum('NHQR,NRHS->NQHS', attention_weights, value)
    return outputs


class DecoderLayer(tf.keras.layers.Layer):
  """The building block that makes the decoder stack of layers, consisting of a 
  self-attention sublayer and a feed-forward sublayer. Processes content stream
  and query stream separately.
  """
  def __init__(self,
               hidden_size,
               num_heads,
               filter_size,
               dropout_rate,
               dropout_rate_attention):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      filter_size: int scalar, the depth of the intermediate dense layer of the
        feed-forward sublayer.
      dropout_rate: float scalar, dropout rate for the Dropout layers.
      dropout_rate_attention: float scalar, dropout rate applied on the
        query-to-reference attention matrix.
    """
    super(DecoderLayer, self).__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate
    self._dropout_rate_attention = dropout_rate_attention

    self._mha = Attention(
        hidden_size, num_heads, dropout_rate_attention)
    self._layernorm_mha = tf.keras.layers.LayerNormalization(epsilon=1e-12)
    self._dropout_mha = tf.keras.layers.Dropout(dropout_rate)

    self._ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
    self._layernorm_ffn = tf.keras.layers.LayerNormalization(epsilon=1e-12)
    self._dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

  def call(self, 
           content_inputs,
           query_inputs,
           position_encoding,
           segment_encoding,
           segment_matrix,
           target_mapping,
           content_mask,
           query_mask,
           content_bias,
           position_bias,
           segment_bias,
           mems=None,
           training=False):
    """Computes the output of the decoder layer.

    Args:
      content_inputs: float tensor of shape [batch_size, q_seq_len,
        hidden_size], input content stream.
      query_inputs: float tensor of shape [batch_size, num_targets,
        hidden_size], input query stream.
      position_encoding: float tensor of shape [batch_size, m_seq_len +
        2 * q_seq_len, hidden_size], encodings of the relative position 
        information.
      segment_encoding: float tensor of shape [2, num_heads, size_per_head],
        embedding vectors of the binary information that whether two positions
        are from the same segment or not.
      segment_matrix: float tensor of shape [batch_size, q_seq_len, m_seq_len +
        q_seq_len], indicator matrix specifying if two positions are from the
        same segment or not.
      target_mapping: float tensor of shape [batch_size, num_targets,
        q_seq_len], one-hot encodings of the indices of prediction targets.
      content_mask: float tensor of shape [batch_size, 1, q_seq_len, m_seq_len +
        q_seq_len], permutation mask for the content stream.
      query_mask: float tensor of shape [batch_size, 1, q_seq_len, m_seq_len +
        q_seq_len], permutation mask for the query stream.
      content_bias: float tensor of shape [num_heads, size_per_head], bias to be
        added to the query sequences.
      position_bias: float tensor of shape [num_heads, size_per_head], bias to
        be added to the query sequences.
      segment_bias: float tensor of shape [num_heads, size_per_head], bias to
        be added to the query sequences.
      mems: (Optional) float tensor of shape [batch_size, m_seq_len, hidden_size
        ], encodings of the memory sequences from the previous block.
      training: (Optional) bool scalar, True if in training mode.

    Returns:
      content_output: float tensor of shape [batch_size, q_seq_len,
        hidden_size], output content stream.
      query_output: float tensor of shape [batch_size, num_targets,
        hidden_size], output query stream.
    """
    content_outputs, query_outputs = self._mha(content_inputs,
                                               query_inputs,
                                               position_encoding,
                                               segment_encoding,
                                               segment_matrix,
                                               target_mapping,
                                               content_mask,
                                               query_mask,
                                               content_bias,
                                               position_bias,
                                               segment_bias,
                                               mems=mems,
                                               training=training)

    content_outputs = self._process_stream(
        content_inputs, content_outputs, training=training)
    query_outputs = self._process_stream(
        query_inputs, query_outputs, training=training)

    return content_outputs, query_outputs

  def _process_stream(self, inputs, outputs, training=False):
    """Processes the results of attention computation of each stream (content or
    query). Applies dropout, layer normalization and residual connections.

    Args:
      inputs: float tensor of shape [batch_size, q_seq_len, hidden_size], inputs
        to `DecoderLayer`.
      outputs: float tensor of shape [batch_size, q_seq_len, hidden_size], 
        results of attention computation.
      training: (Optional) bool scalar, True if in training mode.

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden_size], 
        results of `outputs` after processing.
    """
    outputs = self._dropout_mha(outputs, training=training)
    ffn_inputs = self._layernorm_mha(outputs + inputs)

    outputs = self._ffn(ffn_inputs, training=training)
    outputs = self._dropout_ffn(outputs, training=training)
    outputs = self._layernorm_ffn(outputs + ffn_inputs)
    return outputs 


class TransformerXLModel(tf.keras.layers.Layer):
  """TransformerXL adapted to process two-stream inputs."""
  def __init__(self,
               vocab_size,
               stack_size=6,
               hidden_size=512,
               num_heads=8,
               filter_size=2048,
               mem_len=384,
               reuse_len=256,
               dropout_rate=0.1,
               dropout_rate_attention=0.0,
               tie_biases=True):
    """Constructor.

    Args:
      vocab_size: int scalar, vocabulary size.
      stack_size: int scalar, num of layers in the decoder stack.
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      filter_size: int scalar, the depth of the intermediate dense layer of the
        feed-forward sublayer.
      mem_len: int scalar, num tokens to be cacched.
      reuse_len: int scalar, num of tokens to be reused in the next batch. 
      dropout_rate: float scalar, dropout rate for the Dropout layers.   
      dropout_rate_attention: float scalar, dropout rate applied on the 
        query-to-reference attention matrix. 
      tie_biases: bool scalar, whether to force all layers use the same
        content bias and position bias (True), or create the biases for each
        layer (False).
    """
    super(TransformerXLModel, self).__init__()
    self._vocab_size = vocab_size
    self._stack_size = stack_size
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._mem_len = mem_len
    self._reuse_len = reuse_len
    self._dropout_rate = dropout_rate
    self._dropout_rate_attention = dropout_rate_attention
    self._tie_biases = tie_biases

    self._stack = []
    for i in range(self._stack_size):
      self._stack.append(DecoderLayer(hidden_size,
                                      num_heads,
                                      filter_size,
                                      dropout_rate,
                                      dropout_rate_attention))

  def build(self, inputs_shape):
    """Creates weights of this layer.

    Args:
      inputs_shape: tuple of ints or 1-D int tensor. Not used.
    """
    if self._tie_biases:
      bias_shape = [self._num_heads, self._hidden_size // self._num_heads]
    else:
      bias_shape = [self._stack_size,
                    self._num_heads,
                    self._hidden_size // self._num_heads]

    self._content_bias = self.add_weight(
        'content_bias',
        shape=bias_shape,
        initializer=tf.keras.initializers.RandomNormal(
            mean=0., stddev=self._hidden_size ** -0.5),
        dtype='float32',
        trainable=True)

    self._position_bias = self.add_weight(
        'position_bias',
        shape=bias_shape,
        initializer=tf.keras.initializers.RandomNormal(
            mean=0., stddev=self._hidden_size ** -0.5),
        dtype='float32',
        trainable=True)

    self._segment_bias = self.add_weight(
        'segment_bias',
        initializer=tf.keras.initializers.RandomNormal(
            mean=0., stddev=self._hidden_size ** -0.5),
        shape=bias_shape,
        dtype='float32',
        trainable=True)
    super(TransformerXLModel, self).build(inputs_shape)

  def call(self,
           content_stream,
           query_stream,
           position_encoding,
           segment_encoding,
           segment_matrix,
           target_mapping,
           content_mask,
           query_mask,
           mems=None,
           training=False):
    """Computes query stream output and updates the memory.

    Args:
      content_stream: float tensor of shape [batch_size, q_seq_len,
        hidden_size], input content stream.
      query_stream: float tensor of shape [batch_size, num_targets,
        hidden_size], input query stream.
      position_encoding: float tensor of shape [batch_size, m_seq_len +
        2 * q_seq_len, hidden_size], encodings of the relative position 
        information.
      segment_encoding: float tensor of shape [stack_size, 2, num_heads,
        size_per_head], embedding vectors of the binary information that
        whether two positions are from the same segment or not.      
      segment_matrix: float tensor of shape [batch_size, q_seq_len, m_seq_len +
        q_seq_len], indicator matrix specifying if two positions are from the
        same segment or not.
      target_mapping: float tensor of shape [batch_size, num_targets,
        q_seq_len], one-hot encodings of the indices of prediction targets.
      content_mask: float tensor of shape [batch_size, 1, q_seq_len, m_seq_len +
        q_seq_len], permutation mask for the content stream.
      query_mask: float tensor of shape [batch_size, 1, q_seq_len, m_seq_len +
        q_seq_len], permutation mask for the query stream.
      mems: (Optional) float tensor of shape [batch_size, stack_size, m_seq_len
        , hidden_size], encodings of the memory sequences from the previous
        block.
      training: (Optional) bool scalar, True if in training mode.
 
    Returns:
      query_stream: float tensor of shape [batch_size, num_targets,
        hidden_size], input query stream.      
      new_mems: float tensor of shape [batch_size, stack_size, m_seq_len,
        hidden_size], updated memories.
    """   
    new_mems = []
    mems = [None] * self._stack_size if mems is None else tf.unstack(
        mems, axis=1)

    for i in range(self._stack_size):
      new_mems.append(cache_memory(
          content_stream, mems[i], self._mem_len, self._reuse_len))

      if self._tie_biases:
        content_bias = self._content_bias
        position_bias = self._position_bias
        segment_bias = self._segment_bias
      else:
        content_bias = self._content_bias[i]
        position_bias = self._position_bias[i]
        segment_bias = self._segment_bias[i]

      content_stream, query_stream = self._stack[i](
          content_stream,
          query_stream,
          position_encoding,
          segment_encoding[i],
          segment_matrix,
          target_mapping,
          content_mask,
          query_mask,
          content_bias,
          position_bias,
          segment_bias,
          mems=mems[i],
          training=training)
    new_mems = tf.stack(new_mems, axis=1)
    return query_stream, new_mems


def _compute_attention_mask(perm_mask, m_seq_len, q_seq_len):
  """Compute attention mask for content stream and query stream.

  Args:
    perm_mask: float tensor of shape [batch_size, q_seq_len, q_seq_len]
    m_seq_len: int scalar tensor, memory sequence length.
    q_seq_len: int scalar tensor, query sequence length.

  Returns:
    content_mask: float tensor of shape [batch_size, 1, q_seq_len, m_seq_len +
      q_seq_len], token mask for content stream.
    query_mask: float tensor of shape [batch_size, 1, q_seq_len, m_seq_len +
      q_seq_len], token mask for query stream.
  """
  content_mask = tf.pad(perm_mask * (1 - tf.eye(q_seq_len)), 
      [[0, 0], [0, 0], [m_seq_len, 0]])
  content_mask = content_mask[:, tf.newaxis]

  query_mask = tf.pad(perm_mask,
      [[0, 0], [0, 0], [m_seq_len, 0]])
  query_mask = query_mask[:, tf.newaxis]

  return content_mask, query_mask


def _compute_position_encoding(hidden_size, batch_size, m_seq_len, q_seq_len):
  """Computes position encoding matrix.

  Args:
    hidden_size: int scalar, the hidden size of continuous representation.
    batch_size: int scalar tensor, batch size.
    m_seq_len: int scalar tensor, memory sequence length.
    q_seq_len: int scalar tensor, query sequence length.

  Returns:
    relative_position_encoding: float tensor of shape [batch_size, m_seq_len +
      2 * q_seq_len, hidden_size], the tensor that encodes position 
      information.
  """
  fw_pos_seq = tf.range(m_seq_len + q_seq_len, -q_seq_len, -1.0)
  bw_pos_seq = tf.range(-m_seq_len - q_seq_len, q_seq_len, 1.0)

  fw_position_encoding = get_position_encoding(fw_pos_seq, hidden_size)
  bw_position_encoding = get_position_encoding(bw_pos_seq, hidden_size)
  fw_position_encoding = tf.tile(fw_position_encoding[tf.newaxis],
                                   [batch_size // 2, 1, 1])
  bw_position_encoding = tf.tile(bw_position_encoding[tf.newaxis],
                                   [batch_size // 2, 1, 1])
  relative_position_encoding = tf.concat(
      [fw_position_encoding, bw_position_encoding], axis=0)
  return relative_position_encoding


def _compute_segment_matrix(segment_ids, m_seq_len):
  """Computes the binary matrix indicating whether two positions are from the
  same segment or not.

  Args:
    segment_ids: int tensor of shape [batch_size, q_seq_len], integer matrix
      populated with 0, 1, or 2, where 0 and 1 indicates the corresponding
      position is from the first and the second segment, respectively; and 2
      represents the special token `CLS`.
    m_seq_len: int scalar, memory sequence length.

  Returns:
    segment_matrix: bool tensor of shape [batch_size, q_seq_len, m_seq_len +
      q_seq_len], binary matrix indicating whether two positions are from the
      same segment or not.
  """
  reference_segment_ids = tf.pad(segment_ids, [[0, 0], [m_seq_len, 0]])

  class_index_matrix = tf.logical_or(
      (segment_ids == SEG_ID_CLS)[..., tf.newaxis],
      (reference_segment_ids == SEG_ID_CLS)[:, tf.newaxis])

  segment_matrix = (segment_ids[..., tf.newaxis] == 
                    reference_segment_ids[:, tf.newaxis])

  segment_matrix = tf.logical_or(class_index_matrix, segment_matrix)
  return segment_matrix


def get_position_encoding(pos_seq, hidden_size):
  inverse_frequencies = 1 / (10000 ** (tf.range(0, hidden_size, 2.0) /
      hidden_size))
  position_encoding = tf.einsum('i,j->ij', pos_seq, inverse_frequencies)
  position_encoding = tf.concat([tf.sin(position_encoding),
                                   tf.cos(position_encoding)], axis=1)
  return position_encoding


class XLNetModel(tf.keras.layers.Layer):
  """XLNet model for pretraining as described in 
  https://arxiv.org/abs/1906.08237
  """
  def __init__(self,
               vocab_size=32000,
               stack_size=6,
               hidden_size=512,
               num_heads=8,
               filter_size=2048,
               mem_len=384,
               reuse_len=256,
               dropout_rate=0.1,
               dropout_rate_attention=0.0, 
               tie_biases=False):
    """Constructor.

    Args:
      vocab_size: int scalar, vocabulary size.
      stack_size: int scalar, num of layers in the decoder stack.
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      filter_size: int scalar, the depth of the intermediate dense layer of the
        feed-forward sublayer.
      mem_len: int scalar, num tokens to be cacched.
      reuse_len: int scalar, num of tokens to be reused in the next batch.
      dropout_rate: float scalar, dropout rate for the Dropout layers.   
      dropout_rate_attention: float scalar, dropout rate applied on the 
        query-to-reference attention matrix. 
      tie_biases: bool scalar, whether to force all layers use the same
        content, position and segment bias (True), or create the biases for each
        layer (False).             
    """
    super(XLNetModel, self).__init__()

    self._vocab_size = vocab_size
    self._stack_size = stack_size 
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._mem_len = mem_len
    self._reuse_len = reuse_len
    self._dropout_rate = dropout_rate
    self._dropout_rate_attention = dropout_rate_attention
    self._tie_biases = tie_biases

    self._size_per_head = hidden_size // num_heads

    self._embedding_layer = tf.keras.layers.Embedding(vocab_size, hidden_size)
    self._dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

    self._transformer_xl = TransformerXLModel(
               vocab_size,
               stack_size=stack_size,
               hidden_size=hidden_size,
               num_heads=num_heads,
               filter_size=filter_size,
               mem_len=mem_len,
               reuse_len=reuse_len,
               dropout_rate=dropout_rate,
               dropout_rate_attention=dropout_rate_attention,
               tie_biases=tie_biases)

    self._dense_output = tf.keras.layers.Dense(
          units=hidden_size,
          kernel_initializer=None,
          activation=lambda x: tf.keras.activations.gelu(x, approximate=True))
    self._layernorm_output = tf.keras.layers.LayerNormalization(epsilon=1e-12)

  def build(self, inputs_shape):
    """Creates weights of this layer.

    Args:
      inputs_shape: tuple of ints or 1-D int tensor. Not used.
    """
    self._segment_embedding = self.add_weight(
            'segment_embedding',
            shape=[self._stack_size, 2, self._num_heads, self._size_per_head],
            initializer=tf.keras.initializers.RandomNormal(
                mean=0., stddev=self._hidden_size ** -0.5),
            dtype='float32',
            trainable=True)

    self._mask_embedding = self.add_weight(
            'mask_embedding',
            shape=[1, 1, self._hidden_size],
            initializer=tf.keras.initializers.RandomNormal(
                mean=0., stddev=self._hidden_size ** -0.5),
            dtype='float32',
            trainable=True)

    self._bias_output= self.add_weight(
        "bias_output",
        shape=[self._vocab_size],
        initializer=tf.zeros_initializer())

    super(XLNetModel, self).build(inputs_shape)

  def call(self, 
           inputs,
           segment_ids,
           perm_mask,
           target_mapping,
           masked_tokens,
           mems=None):
    """Computes the output query stream and update the memories.

    Args:
      inputs: int tensor of shape [batch_size, q_seq_len], token ids of input
        query sequences of the current batch.
      segment_ids: int tensor of shape [batch_size, q_seq_len], matrix populated
        with 0, 1, or 2, which indicates the corresponding index in `inputs`
        is from the first and second sequence segment, or is the `CLS` token.
      perm_mask: int tensor of shape [batch_size, q_seq_len, q_seq_len], binary
        matrix where 1 means the corresponding position cannot be attended to.
      target_mapping: float tensor of shape [batch_size, num_targets,
        q_seq_len], one-hot encodings of the indices of prediction targets.
      masked_tokens: bool tensor of shape [batch_size, q_seq_len], binary matrix
        where 1 means the corresponding position is the prediction target. 
      mems: (Optional) float tensor of shape [batch_size, stack_size, m_seq_len
        , hidden_size], encodings of the memory sequences from the previous
        block.

    Returns:
      logits: float tensor of shape [batch_size, num_targets, vocab_size],
        logits over vocabulary.
      new_mems: float tensor of shape [batch_size, stack_size, m_seq_len,
        hidden_size], updated memories.
      query_stream: float tensor of shape [batch_size, num_targets,
        hidden_size], input query stream.      
    """
    batch_size = tf.shape(inputs)[0]
    q_seq_len = tf.shape(inputs)[1]
    m_seq_len = 0 if mems is None else tf.shape(mems[0])[1]

    content_mask, query_mask = _compute_attention_mask(
        perm_mask, m_seq_len, q_seq_len)
    relative_position_encoding = self._dropout_layer(
        _compute_position_encoding(
            self._hidden_size, batch_size, m_seq_len, q_seq_len))
    segment_matrix = _compute_segment_matrix(segment_ids, m_seq_len)

    content_stream = self._dropout_layer(self._embedding_layer(inputs))
    query_stream = self._dropout_layer(tf.broadcast_to(self._mask_embedding,
        tf.shape(target_mapping)))

    query_stream, new_mems = self._transformer_xl(
        content_stream,
        query_stream,
        relative_position_encoding,
        self._segment_embedding,
        segment_matrix,
        target_mapping,
        content_mask,
        query_mask,
        mems=mems)

    outputs = self._layernorm_output(self._dense_output(query_stream))
    logits = tf.einsum("NPD,VD->NPV", outputs, self._embedding_layer.weights[0]
        ) + self._bias_output

    return logits, new_mems, query_stream

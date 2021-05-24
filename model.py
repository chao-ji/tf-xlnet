"""Defines TransformerXL model in tf.keras.API."""
import tensorflow as tf
import math
from commons.layers import Projection
from commons.layers import FeedForwardNetwork
from commons.beam_search import NEG_INF


_SEG_ID_CLS = 2


def gelu(x):
  return tf.keras.activations.gelu(x, approximate=True)

def _cache_memory(current_state, previous_state, memory_length, reuse_length=0):
  """Caches hidden states into memory.

  Arguments:
    current_state: `Tensor`, the current state.
    previous_state: `Tensor`, the previous state.
    memory_length: `int`, the number of tokens to cache.
    reuse_length: `int`, the number of tokens in the current batch to be cached
      and reused in the future.

  Returns:
    A `Tensor`, representing the cached state with stopped gradients.

  """
  if memory_length is None or memory_length == 0:
    return None
  else:
    if reuse_length > 0:
      current_state = current_state[:, :reuse_length, :]

    if previous_state is None:
      new_mem = current_state[:, -memory_length:, :]
    else:
      new_mem = tf.concat( 
          [previous_state, current_state], 1)[:, -memory_length:, :]

  return tf.stop_gradient(new_mem)


def rel_shift(inputs):
  shape = tf.shape(inputs)
  padded = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [1, 0]])
  reshaped = tf.reshape(padded, [shape[0], shape[1], shape[3] + 1, shape[2]])
  sliced = reshaped[:, :, 1:]
  outputs = tf.reshape(sliced, shape)
  return outputs


class Attention(tf.keras.layers.Layer):
  """Multi-headed attention layer used in TransformerXL model. The content and
  position bias will be provided by the caller.
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
           content_stream,
           query_stream,
           positional_encoding,
           segment_matrix,
           segment_encoding,
           segment_bias,
           content_mask,
           query_mask,
           target_mapping,
           content_bias,
           position_bias,
           state=None):
    """

    Args:
      content_stream: float tensor of shape [batch_size, q_seq_len, hidden_size]
      query_stream: float tensor of shape [batch_size, num_predictions, hidden_size]
      positional_encoding: float tensor of shape [batch_size, q_seq_len * 2, hidden_size]
      segment_matrix: float tensor of shape [batch_size, q_seq_len, q_seq_len]
      segment_encoding: float tensor of shape [num_segments, num_heads, size_per_head]
      segment_bias: float tensor of shape [num_heads, size_per_head]
      content_mask: float tensor of shape [batch_size, 1, q_seq_len, q_seq_len]
      target_mapping: float tensor of shape [batch_size, num_predictions, q_seq_len]
      content_bias: float tensor of shape [num_heads, size_per_head]
      position_bias: float tensor of shape [num_heads, size_per_head]


    Returns:
      content_output: float tensor of shape [batch_size, q_seq_len, hidden_size]
      query_output: float tensor of shape [batch_size, num_predictions, hidden_size]
    """
    print('content_stream', content_stream.shape)
    print('query_stream', query_stream.shape)
    print('positional_encoding', positional_encoding.shape, positional_encoding.numpy().sum())
    print('segment_matrix', segment_matrix.shape)
    print('segment_encoding', segment_encoding.shape)
    print('segment_bias', segment_bias.shape)
    print('content_mask', content_mask.shape)
    print('query_mask', query_mask.shape)
    print('target_mapping', target_mapping.shape)
    print('content_bias', content_bias.shape)
    print('position_bias', position_bias.shape)
    print(state)

    if state is not None:
      content_and_memory_stream = tf.concat([state, content_stream], 1)
    else:
      content_and_memory_stream = content_stream

    # [batch_size, q_seq_len, num_heads, size_per_head]
    query = self._dense_layer_query(content_stream)

    # [batch_size, q_seq_len, num_heads, size_per_head]
    key_content = self._dense_layer_key_content(content_and_memory_stream)

    # [batch_size, q_seq_len, num_heads, size_per_head]
    value = self._dense_layer_value(content_and_memory_stream)

    # [batch_size, 2 * seq_len, num_heads, size_per_head]
    key_position = self._dense_layer_key_position(positional_encoding)

    # [batch_size, q_seq_len, num_heads, size_per_head]
    content_output = self._compute_attention(query,
                                             key_content,
                                             value,
                                             key_position,
                                             content_bias,
                                             position_bias,
                                             segment_matrix,
                                             segment_encoding,
                                             segment_bias,
                                             content_mask)

    # [batch_size, q_seq_len, hidden_size]
    content_output = self._dense_layer_output(content_output)

    # [batch_size, num_predictions, num_heads, size_per_head]
    query = self._dense_layer_query(query_stream)

    # [batch_size, q_seq_len, num_heads, size_per_head]
    query = tf.einsum('NPHS,NPQ->NQHS', query, target_mapping)

    # [batch_size, q_seq_len, num_heads, size_per_head]
    query_output = self._compute_attention(query,
                                           key_content,
                                           value,
                                           key_position,
                                           content_bias,
                                           position_bias,
                                           segment_matrix,
                                           segment_encoding,
                                           segment_bias,
                                           query_mask) 

    # [batch_size, num_predictions, num_heads, size_per_head]
    query_output = tf.einsum('blnd,bml->bmnd', query_output, target_mapping)

    # [batch_size, num_predictions, hidden_size]
    query_output = self._dense_layer_output(query_output)

    return content_output, query_output

  def _compute_attention(self,
                         query,
                         key_content,
                         value,
                         key_position,
                         content_bias,
                         position_bias,
                         segment_matrix,
                         segment_encoding,
                         segment_bias,
                         token_mask):
    """
    Args:
      query: float tensor of shape [batch_size, q_seq_len, num_heads, size_per_head]
      key_content: float tensor of shape [batch_size, q_seq_len, num_heads, size_per_head] 
      value: float tensor of shape [batch_size, q_seq_len, num_heads, size_per_head] 
      key_position: float tensor of shape [batch_size, 2 * q_seq_len, num_heads, size_per_head]
      content_attention_bias: float tensor of shape [num_heads, size_per_head]
      positional_attention_bias: float tensor of shape [num_heads, size_per_head]
      segment_matrix: float tensor of shape [batch_size, q_seq_len, q_seq_len] 
      segment_encoding: float tensor of shape [num_segments, num_heads, size_per_head]
      segment_attention_bias: float tensor of shape [num_heads, size_per_head]
      token_mask: float tensor of shape [batch_size, 1, q_seq_len, q_seq_len] 

    """
    print()
    print('_compute_attention')
    print('query', query.shape)
    print('key_content', key_content.shape)
    print('value', value.shape)
    print('key_position', key_position.shape)
    print('content_bias', content_bias.shape)
    print('position_bias', position_bias.shape)
    print('segment_matrix', segment_matrix.shape)
    print('segment_encoding', segment_encoding.shape)
    print('segment_bias', segment_bias.shape)
    print('token_mask', token_mask.shape)
    print('---------')

    # [batch_size, num_heads, q_seq_len, q_seq_len]
    content = tf.einsum('NQHS,NRHS->NHQR',
        query + content_bias, key_content)
    print('content_attention', content.shape)

    # [batch_size, num_heads, q_seq_len, 2 * q_seq_len]
    position = tf.einsum('NQHS,NRHS->NHQR',
        query + position_bias, key_position)
    print('position_attention', position.shape)

    # [batch_size, num_heads, q_seq_len, q_seq_len]
    position = rel_shift(position)[
        ..., 1:1 + tf.shape(content)[3]]
    print('position_attention', position.shape)

    # [batch_size, num_heads, q_seq_len, num_segments]
    segment = tf.einsum("bind,snd->bnis",
                                   query + segment_bias,
                                   segment_encoding)
    print('segment_attention', segment.shape, position.shape)

    target_shape = tf.shape(position)

    # [batch_size, num_heads, q_seq_len, q_seq_len]    
    segment = tf.where(
        tf.broadcast_to(tf.expand_dims(segment_matrix, 1), target_shape),
        tf.broadcast_to(segment[:, :, :, 1:], target_shape),
        tf.broadcast_to(segment[:, :, :, :1], target_shape))
    print('segment_attention', segment.shape)

    attention_weights = tf.multiply(
        content + position + segment, 
        1.0 / math.sqrt(float(self._size_per_head)))

    attention_weights += token_mask * NEG_INF
    attention_weights = tf.nn.softmax(attention_weights, axis=3)
    attention_weights = self._attention_dropout_layer(attention_weights)

    outputs = tf.einsum('NHQR,NRHS->NQHS', attention_weights, value)
    return outputs


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               hidden_size,
               num_heads,
               filter_size,
               dropout_rate,
               dropout_rate_attention,
               reuse_biases=False):

    super(DecoderLayer, self).__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate
    self._dropout_rate_attention = dropout_rate_attention
    self._reuse_biases = reuse_biases



    self._mha = Attention(
        hidden_size, num_heads, dropout_rate_attention)
    self._layernorm_mha = tf.keras.layers.LayerNormalization(epsilon=1e-12)
    self._dropout_mha = tf.keras.layers.Dropout(dropout_rate)

    self._ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
    self._layernorm_ffn = tf.keras.layers.LayerNormalization(epsilon=1e-12)
    self._dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

  def call(self, 
           content_stream,
          query_stream,
          positional_encoding,
          segment_matrix,
          segment_encoding,
          segment_bias,
          content_mask,
          query_mask,
          target_mapping,
          content_bias,
          position_bias,
          state=None):
    attention_output = self._mha(content_stream,
                                  query_stream,
          positional_encoding,
          segment_matrix,
          segment_encoding,
          segment_bias,
          content_mask,
          query_mask,
          target_mapping,
          content_bias=content_bias,
          position_bias=position_bias,
          state=state)


    attention_streams = attention_output

    input_streams = [content_stream, query_stream]
    output = []
    for attention_stream, input_stream in zip(attention_streams, input_streams):
      attention_stream = self._dropout_mha(attention_stream)
      attention_stream = self._layernorm_mha(attention_stream + input_stream)
      inner_output = self._ffn(attention_stream)
      inner_output = self._dropout_ffn(inner_output)
      layer_output = self._layernorm_ffn(inner_output + attention_stream)

      output.append(layer_output)

    return output


class TransformerXLModel(tf.keras.layers.Layer):
  def __init__(self,
               adaptive_embedding,
               vocab_size,
               cutoffs=None,
               stack_size=6,
               hidden_size=512,
               num_heads=8,
               filter_size=2048,
               memory_length=384,
               reuse_length=256,
               dropout_rate=0.1,
               dropout_rate_attention=0.0,
               tie_biases=True):

    super(TransformerXLModel, self).__init__()
    self._adaptive_embedding = adaptive_embedding
    self._vocab_size = vocab_size
    self._cutoffs = cutoffs
    self._stack_size = stack_size
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate
    self._dropout_rate_attention = dropout_rate_attention
    self._tie_biases = tie_biases


    self._memory_length = memory_length 
    self._reuse_length = reuse_length

    if tie_biases:
      bias_shape = [num_heads, hidden_size // num_heads]
    else:
      bias_shape = [stack_size, num_heads, hidden_size // num_heads]


    self.content_attention_bias = self.add_weight(
        'content_attention_bias',
        shape=bias_shape,
        dtype='float32')

    self.positional_attention_bias = self.add_weight(
        'positional_attention_bias',
        shape=bias_shape,
        dtype='float32')

    self.segment_attention_bias = self.add_weight(
        'segment_attention_bias',
        shape=bias_shape,
        dtype='float32')


    self._embeddings_dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    self._positional_encoding_dropout_layer = tf.keras.layers.Dropout(
        dropout_rate)
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    self._stack = []



    for i in range(self._stack_size):
      self._stack.append(DecoderLayer(hidden_size,
                                      num_heads,
                                      filter_size,
                                      dropout_rate,
                                      dropout_rate_attention))
  def call(self, content_stream, query_stream, positional_encoding, segment_matrix, segment_embedding, content_mask, query_mask, target_mapping, state=None):
    
    new_mems = []
    if state is None:
      state = [None] * self._stack_size
    for i in range(self._stack_size):
      new_mems.append(_cache_memory(content_stream, state[i], self._memory_length, self._reuse_length))

      segment_encoding = segment_embedding[i]


      if self._tie_biases:
        segment_attention_bias = self.segment_attention_bias
        content_attention_bias = self.content_attention_bias
        positional_attention_bias = self.positional_attention_bias
      
      else:
        segment_attention_bias = self.segment_attention_bias[i]
        content_attention_bias = self.content_attention_bias[i]
        positional_attention_bias = self.positional_attention_bias[i]
      content_stream, query_stream = self._stack[i](content_stream,
          query_stream,
          positional_encoding,
          segment_matrix,
          segment_encoding,
          segment_attention_bias,
          content_mask,
          query_mask,
          target_mapping,
          content_attention_bias,
          positional_attention_bias,
          state=state[i])
      print('\n\n\n\n')
    return query_stream, new_mems





def _compute_attention_mask(
    input_mask,
    permutation_mask,
    attention_type,
    seq_length,
    memory_length,
    batch_size,
    dtype=tf.float32):
  attention_mask = None
  # `1` values mean do not attend to this position.
  if attention_type == "uni":
    causal_attention_mask = _create_causal_attention_mask(
        seq_length=seq_length,
        memory_length=memory_length,
        dtype=dtype)
    causal_attention_mask = causal_attention_mask[None, None, :, :]
    # `causal_attention_mask`: [1, 1, S, S + M]

  # input_mask: [B, S]
  # permutation_mask: [B, S, S]
  if input_mask is not None and permutation_mask is not None:
    data_mask = input_mask[:, None, :] + permutation_mask

  elif input_mask is not None and permutation_mask is None:
    data_mask = input_mask[:, None, :]
  elif input_mask is None and permutation_mask is not None:
    data_mask = permutation_mask
  else:
    data_mask = None


  # data_mask: [B, S, S] or [B, 1, S]

  if data_mask is not None:
    # All positions within state can be attended to.
    state_mask = tf.zeros([batch_size, tf.shape(data_mask)[1], memory_length],
                          dtype=dtype)
    # state_mask: [B, 1, M] or [B, S, M]
    data_mask = tf.concat([state_mask, data_mask], 2)
    # data_mask: [B, 1, S + M] or [B, S, S + M]

    if attention_type == "uni":
      attention_mask = causal_attention_mask + data_mask[:, None, :, :]
    else:
      attention_mask = data_mask[:, None, :, :]

  # Construct the content attention mask.
  if attention_mask is not None:
    attention_mask = tf.cast(attention_mask > 0, dtype=dtype)

    non_tgt_mask = -tf.eye(seq_length, dtype=dtype)
    non_tgt_mask = tf.concat(
        [tf.zeros([seq_length, memory_length], dtype=dtype),
         non_tgt_mask], axis=-1)
    content_attention_mask = tf.cast(
        (attention_mask + non_tgt_mask[None, None, :, :]) > 0,
        dtype=dtype)
  else:
    content_attention_mask = None

  return attention_mask, content_attention_mask



def _compute_positional_encoding(
    attention_type,
    position_encoding_layer,
    hidden_size,
    batch_size,
    total_length,
    seq_length,
    clamp_length,
    bi_data,
    dtype=tf.float32):
  freq_seq = tf.range(0, hidden_size, 2.0)
  if dtype is not None and dtype != tf.float32:
    freq_seq = tf.cast(freq_seq, dtype=dtype)

  if attention_type == "bi":
    beg, end = total_length, -seq_length
  elif attention_type == "uni":
    beg, end = total_length, -1
  else:
    raise ValueError("Unknown `attention_type` {}.".format(attention_type))

  if bi_data:
    forward_position_sequence = tf.range(beg, end, -1.0)
    backward_position_sequence = tf.range(-beg, -end, 1.0)

    if dtype is not None and dtype != tf.float32:
      forward_position_sequence = tf.cast(forward_position_sequence,
                                          dtype=dtype)
      backward_position_sequence = tf.cast(backward_position_sequence,
                                           dtype=dtype)

    if clamp_length > 0:
      forward_position_sequence = tf.clip_by_value(
          forward_position_sequence,
          -clamp_length,
          clamp_length)
      backward_position_sequence = tf.clip_by_value(
          backward_position_sequence,
          -clamp_length,
          clamp_length)

    if batch_size is not None:
      forward_positional_encoding = position_encoding_layer(
          forward_position_sequence, batch_size // 2)
      backward_positional_encoding = position_encoding_layer(
          backward_position_sequence, batch_size // 2)
    else:
      forward_positional_encoding = position_encoding_layer(
          forward_position_sequence, None)
      backward_positional_encoding = position_encoding_layer(
          backward_position_sequence, None)

    relative_position_encoding = tf.concat(
        [forward_positional_encoding, backward_positional_encoding], axis=0)
  else:
    forward_position_sequence = tf.range(beg, end, -1.0)
    if dtype is not None and dtype != tf.float32:
      forward_position_sequence = tf.cast(
          forward_position_sequence, dtype=dtype)
    if clamp_length > 0:
      forward_position_sequence = tf.clip_by_value(
          forward_position_sequence,
          -clamp_length,
          clamp_length)

    relative_position_encoding = position_encoding_layer(
        forward_position_sequence, batch_size)
  return relative_position_encoding


def _compute_segment_matrix(
    segment_ids,
    memory_length,
    batch_size,
    use_cls_mask):

  if segment_ids is None:
    return None

  memory_padding = tf.zeros([batch_size, memory_length], dtype=tf.int32)
  padded_segment_ids = tf.concat([memory_padding, segment_ids], 1)
  # segment_ids: [B, S]
  # padded_segment_ids: [B, S + M]

  if use_cls_mask:
    # `1` indicates not in the same segment.
    # Target result: [B, S, S + M]

    # segment_ids: [B, S]
    # padded_segment_ids: [B, S + M]
    broadcasted_segment_class_indices = (
        tf.equal(segment_ids,
                 tf.constant([_SEG_ID_CLS]))[:, :, None])

    broadcasted_padded_class_indices = (
        tf.equal(
            padded_segment_ids,
            tf.constant([_SEG_ID_CLS]))[:, None, :])

    class_index_matrix = tf.logical_or(broadcasted_segment_class_indices,
                                       broadcasted_padded_class_indices)

    segment_matrix = tf.equal(segment_ids[:, :, None],
                              padded_segment_ids[:, None, :])
    segment_matrix = tf.logical_or(class_index_matrix, segment_matrix)
  else:
    segment_matrix = tf.logical_not(
        tf.equal(segment_ids[:, :, None], padded_segment_ids[:, None, :]))
  return segment_matrix



class RelativePositionEncoding(tf.keras.layers.Layer):
  def __init__(self, hidden_size, **kwargs):
    super(RelativePositionEncoding, self).__init__(**kwargs)
    self._hidden_size = hidden_size
    self._inv_freq = 1.0 / (10000.0**(
        tf.range(0, self._hidden_size, 2.0) / self._hidden_size))

  def call(self, pos_seq, batch_size=None):
    """Implements call() for the layer.

    Arguments:
      pos_seq: A 1-D `Tensor`
      batch_size: The optionally provided batch size that tiles the relative
        positional encoding.

    Returns:
      The relative positional encoding of shape:
        [batch_size, len(pos_seq), hidden_size] if batch_size is provided, else
        [1, len(pos_seq), hidden_size].
    """
    sinusoid_input = tf.einsum("i,d->id", pos_seq, self._inv_freq)
    relative_position_encoding = tf.concat([tf.sin(sinusoid_input),
                                            tf.cos(sinusoid_input)], -1)
    relative_position_encoding = relative_position_encoding[None, :, :]
    if batch_size is not None:
      relative_position_encoding = tf.tile(relative_position_encoding,
                                           [batch_size, 1, 1])
    return relative_position_encoding


class XLNetModel(tf.keras.layers.Layer):
  def __init__(self,
               tie_biases,
               vocab_size=32000,
               num_layers=6,
               num_heads=8,
               hidden_size=512,
               filter_size=2048,
               memory_length=384,
               reuse_length=256,
               dropout_rate=0.1,
               clamp_length=-1,
               bi_data=True,
               use_cls_mask=True,
               attention_type="bi"): 
    super(XLNetModel, self).__init__()

    self._tie_biases = tie_biases

    adaptive_embedding = False
    self._vocab_size = vocab_size

    self._num_layers = num_layers
    self._num_heads = num_heads
    self._size_per_head = hidden_size // num_heads
    self._hidden_size = hidden_size
    self._bi_data = bi_data
    self._dropout_rate = dropout_rate

    self._use_cls_mask = use_cls_mask

    self._attention_type = attention_type
    self._clamp_length = clamp_length
    self.embedding_dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    self._embedding_layer = tf.keras.layers.Embedding(vocab_size, hidden_size)
    self._dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    self.position_encoding = RelativePositionEncoding(hidden_size)

    self._segment_embedding = self.add_weight(
            "seg_embed",
            shape=[num_layers, 2, num_heads,
                   self._size_per_head],
            dtype=tf.float32)

    self._mask_embedding = self.add_weight(
            "mask_emb/mask_emb",
            shape=[1, 1, hidden_size],
            dtype=tf.float32)

    self._transformer_xl = TransformerXLModel(adaptive_embedding,
               vocab_size,
               cutoffs=None,
               stack_size=num_layers,
               hidden_size=hidden_size,
               num_heads=num_heads,
               filter_size=filter_size,
               memory_length=memory_length,
               reuse_length=reuse_length,
               dropout_rate=dropout_rate,
               dropout_rate_attention=0.0,
               tie_biases=tie_biases)

  def call(self, 
           input_ids,
           segment_ids,
           input_mask,
               state,
               permutation_mask,
               target_mapping,
               masked_tokens):


    batch_size = tf.shape(input_ids)[0]
    seq_length = tf.shape(input_ids)[1]

    memory_length = 0
    total_length = memory_length + seq_length

    query_attention_mask, content_attention_mask = _compute_attention_mask(
        input_mask=input_mask,
        permutation_mask=permutation_mask,
        attention_type=self._attention_type,
        seq_length=seq_length,
        memory_length=memory_length,
        batch_size=batch_size,
        dtype=tf.float32)

    relative_position_encoding = _compute_positional_encoding(
        attention_type=self._attention_type,
        position_encoding_layer=self.position_encoding,
        hidden_size=self._hidden_size,
        batch_size=batch_size,
        total_length=total_length,
        seq_length=seq_length,
        clamp_length=self._clamp_length,
        bi_data=self._bi_data,
        dtype=tf.float32)
    segment_embedding = self._segment_embedding
    segment_matrix = _compute_segment_matrix(
          segment_ids=segment_ids,
          memory_length=memory_length,
          batch_size=batch_size,
          use_cls_mask=self._use_cls_mask)

    word_embeddings = self._embedding_layer(input_ids)

    content_stream = self._dropout(word_embeddings)

    masked_token_embedding = tf.tile(
            self._mask_embedding,
            [batch_size, tf.shape(target_mapping)[1], 1])

    query_stream = self._dropout(masked_token_embedding)

    return self._transformer_xl(content_stream, query_stream, relative_position_encoding, segment_matrix, segment_embedding, content_attention_mask, query_attention_mask, target_mapping, None)




class LMLossLayer(tf.keras.layers.Layer):
  """Layer computing cross entropy loss for language modeling."""

  def __init__(self,
               vocab_size,
               hidden_size,
               initializer,
               tie_weight=False,
               bi_data=True,
               use_one_hot=False,
               use_proj=False,
               **kwargs):
    super(LMLossLayer, self).__init__(**kwargs)
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.initializer = initializer

    self.tie_weight = tie_weight
    self.bi_data = bi_data
    self.use_one_hot = use_one_hot
    self.use_proj = use_proj

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    if self.use_proj:
      self.proj_layer = tf.keras.layers.Dense(
          units=self.hidden_size,
          kernel_initializer=self.initializer,
          activation=gelu,
          name="lm_projection/dense")
      self.proj_layer_norm = tf.keras.layers.LayerNormalization(
          axis=-1, epsilon=1e-12, name="lm_projection/LayerNorm")
    if not self.tie_weight:
      self.softmax_w = self.add_weight(
          "weight",
          shape=[self.vocab_size, self.hidden_size],
          initializer=self.initializer)

    self.softmax_b = self.add_weight(
        "bias", shape=[self.vocab_size], initializer=tf.zeros_initializer())

    super(LMLossLayer, self).build(unused_input_shapes)

  def call(self, hidden, target, lookup_table, target_mask):
    """Implements call() for the layer."""
    if self.use_proj:
      hidden = self.proj_layer_norm(self.proj_layer(hidden))
    if self.tie_weight:
      logits = tf.einsum("ibd,nd->ibn", hidden, lookup_table) + self.softmax_b
    else:
      logits = tf.einsum("ibd,nd->ibn", hidden, self.softmax_w) + self.softmax_b

    if self.use_one_hot:
      one_hot_target = tf.one_hot(target, self.vocab_size, dtype=logits.dtype)
      loss = -tf.reduce_sum(tf.nn.log_softmax(logits) * one_hot_target, -1)
    else:
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=target, logits=logits)

    total_loss = tf.reduce_sum(loss * target_mask) / tf.reduce_sum(target_mask)

    return total_loss, logits




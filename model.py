"""Defines XLNet model in tf.keras.API."""
import tensorflow as tf

from commons.beam_search import NEG_INF
from commons.layers import FeedForwardNetwork
from commons.layers import RelativeAttention

from model_utils import cache_memory
from model_utils import compute_attention_mask
from model_utils import get_position_encoding_xlnet
from model_utils import compute_segment_matrix
from model_utils import get_position_encoding


NEG_INF = -1e30

class DecoderLayer(tf.keras.layers.Layer):
  """TransformerXL is created by stacking up multiple copies of this layer. Each
  layer contains a self-attention sublayer followed by a feed-forward sublayer.
  Takes content stream (and optionally query stream) as input sequences.
  """
  def __init__(self,
               hidden_size,
               num_heads,
               filter_size,
               dropout_rate=0.0,
               dropout_rate_attention=0.0,
               filter_activation=tf.nn.relu):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      filter_size: int scalar, the depth of the intermediate dense layer of the
        feed-forward sublayer.
      dropout_rate: (Optional) float scalar, dropout rate for the Dropout layers
        . Defaults to 0.
      dropout_rate_attention: (Optional) float scalar, dropout rate applied on
        the query-to-reference attention matrix. Defaults to 0.
      filter_activation: (Optional) callable or string, activation function of
        the filter dense layer. Defaults to ReLU.
    """
    super(DecoderLayer, self).__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate
    self._dropout_rate_attention = dropout_rate_attention
    self._filter_activation = filter_activation

    self._mha = RelativeAttention(
        hidden_size, num_heads, dropout_rate_attention, for_xlnet=True)
    self._layernorm_mha = tf.keras.layers.LayerNormalization(epsilon=1e-12)
    self._dropout_mha = tf.keras.layers.Dropout(dropout_rate)

    self._ffn = FeedForwardNetwork(
        hidden_size, filter_size, dropout_rate, filter_activation)
    self._layernorm_ffn = tf.keras.layers.LayerNormalization(epsilon=1e-12)
    self._dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

  def call(self,
           content_stream,
           content_mask,
           context,
           position_encoding,
           content_bias,
           position_bias,
           segment_encoding,
           segment_matrix,
           segment_bias,
           query_stream=None,
           query_mask=None,
           target_mapping=None,
           training=False):
    """Computes the output of the decoder layer.

    Args:
      content_stream: float tensor of shape [batch_size, q_seq_len, hidden_size]
        , the query sequences for TransformerXL or the content stream for
        pre-training XLNet.
      content_mask: float tensor of shape [batch_size, 1, q_seq_len, c_seq_len],
        token mask for content stream.
      context: float tensor of shape [batch_size, c_seq_len, hidden_size], the
        context sequences to which the query sequences will attend.
      position_encoding: float tensor of shape [batch_size, r_seq_len,
        hidden_size], the position encoding for the context sequences.
      content_bias: float tensor of shape [num_heads, size_per_head], content
        bias.
      position_bias: float tensor of shape [num_heads, size_per_head], position
        bias.
      segment_encoding: float tensor of shape [2, num_heads, size_per_head],
        embedding vectors of the binary information that whether two positions
        are from the same segment or not.
      segment_matrix: bool tensor of shape [batch_size, q_seq_len, c_seq_len],
        binary matrix indicating whether two positions are from the same segment
        or not.
      segment_bias: float tensor of shape [num_heads, size_per_head], segment
        bias.
      query_stream: (Optional) float tensor of shape [batch_size, num_predict,
        hidden_size], the query stream for pre-training XLNet. Defaults to None.
      query_mask: (Optional) float tensor of shape [batch_size, 1, q_seq_len,
        c_seq_len], token mask for query stream. Defaults to None.
      target_mapping: (Optional) float tensor of shape [batch_size, num_predict,
        q_seq_len], one-hot encodings of the indices of prediction targets.
        Defaults to None.
      training: (Optional) bool scalar, True if in training mode. Defaults to
        False.

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden_size], for
        single stream input; or a tuple of two tensors of shape [batch_size,
        q_seq_len, hidden_size] and [batch_size, num_predict, hidden_size].
    """
    kwargs = {'content_stream': content_stream,
              'content_mask': content_mask,
              'context': context,
              'position_encoding': position_encoding,
              'content_bias': content_bias,
              'position_bias': position_bias,
              'segment_encoding': segment_encoding,
              'segment_matrix': segment_matrix,
              'segment_bias': segment_bias,
              'query_stream': query_stream,
              'query_mask': query_mask,
              'target_mapping': target_mapping,
              'training': training}

    if query_stream is not None:
      content_outputs, query_outputs = self._mha(**kwargs)
      content_outputs = self._process_stream(
          content_stream, content_outputs, training=training)
      query_outputs = self._process_stream(
          query_stream, query_outputs, training=training)

      return content_outputs, query_outputs
    else:
      content_outputs = self._mha(**kwargs)
      content_outputs = self._process_stream(
          content_stream, content_outputs, training=training)
      return content_outputs

  def _process_stream(self, inputs, outputs, training=False):
    """Processes the results of attention computation of each stream (content or
    query). Applies dropout, layer normalization and residual connections.

    Args:
      inputs: float tensor of shape [batch_size, q_seq_len, hidden_size], inputs
        to `DecoderLayer`.
      outputs: float tensor of shape [batch_size, q_seq_len, hidden_size],
        results of attention computation.
      training: (Optional) bool scalar, True if in training mode. Defaults to
        False.

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
  """TransformerXL adapted to optionally process query stream on top of
  content stream.
  """
  def __init__(self,
               vocab_size,
               stack_size=6,
               hidden_size=512,
               num_heads=8,
               filter_size=2048,
               mem_len=0,
               reuse_len=0,
               dropout_rate=0.0,
               dropout_rate_attention=0.0,
               tie_biases=False,
               filter_activation=tf.nn.relu):
    """Constructor.

    Args:
      vocab_size: int scalar, vocabulary size.
      stack_size: (Optional) int scalar, num of layers in the decoder stack.
        Defaults to 6.
      hidden_size: (Optional) int scalar, the hidden size of continuous
        representation. Defaults to 512.
      num_heads: (Optional) int scalar, num of attention heads. Defaults to 8.
      filter_size: (Optional) int scalar, the depth of the intermediate dense
        layer of the feed-forward sublayer. Defaults to 2048.
      mem_len: (Optional) int scalar, num tokens to be cached. Defaults to 0.
      reuse_len: (Optional) int scalar, num of tokens to be reused in the next
        batch. Defaults to 0.
      dropout_rate: (Optional) float scalar, dropout rate for Dropout layers.
        Defaults to 0.
      dropout_rate_attention: (Optional) float scalar, dropout rate applied on
        the query-to-reference attention matrix. Defaults to 0.
      tie_biases: (Optional) bool scalar, whether to force all layers use the
        same content bias and position bias (True), or create the biases for
        each layer (False). Defaults to False.
      filter_activation: (Optional) callable or string, activation function of
        the filter dense layer. Defaults to ReLU.
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
    self._filter_activation = filter_activation

    self._size_per_head = hidden_size // num_heads

    self._stack = []
    for i in range(self._stack_size):
      self._stack.append(DecoderLayer(hidden_size,
                                      num_heads,
                                      filter_size,
                                      dropout_rate,
                                      dropout_rate_attention,
                                      filter_activation))

  def build(self, inputs_shape):
    """Creates weights of this layer.

    Args:
      inputs_shape: tuple of ints or 1-D int tensor. Not used.
    """
    if self._tie_biases:
      bias_shape = [self._num_heads, self._size_per_head]
    else:
      bias_shape = [self._stack_size, self._num_heads, self._size_per_head]

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

    self._segment_encoding = self.add_weight(
            'segment_encoding',
            shape=[self._stack_size, 2, self._num_heads, self._size_per_head],
            initializer=tf.keras.initializers.RandomNormal(
                mean=0., stddev=self._hidden_size ** -0.5),
            dtype='float32',
            trainable=True)
    super(TransformerXLModel, self).build(inputs_shape)

  def call(self,
           content_stream,
           content_mask,
           position_encoding,
           segment_matrix,
           query_stream=None,
           query_mask=None,
           mems=None,
           target_mapping=None,
           training=False):
    """Computes query stream output and updates the memory.

    Args:
      content_stream: float tensor of shape [batch_size, q_seq_len,
        hidden_size], input content stream.
      content_mask: float tensor of shape [batch_size, 1, q_seq_len, m_seq_len +
        q_seq_len], permutation mask for the content stream.
      position_encoding: float tensor of shape [batch_size, m_seq_len +
        2 * q_seq_len, hidden_size], encodings of the relative position
        information.
      segment_matrix: float tensor of shape [batch_size, q_seq_len, m_seq_len +
        q_seq_len], indicator matrix specifying if two positions are from the
        same segment or not.
      query_stream: (Optional) float tensor of shape [batch_size, num_predict,
        hidden_size], input query stream. Defaults to None.
      query_mask: (Optional) float tensor of shape [batch_size, 1, q_seq_len,
        m_seq_len + q_seq_len], permutation mask for the query stream. Defaults
        to None.
      mems: (Optional) float tensor of shape [batch_size, stack_size, m_seq_len
        , hidden_size], encodings of the memory sequences from the previous
        block. Defaults to None.
      target_mapping: (Optional) float tensor of shape [batch_size, num_predict,
        q_seq_len], where `target_mapping[b, i]` is the one-hot encoding of the
        index of the prediction target for the `i` prediction task (out of
        `num_predict`). May be zero-padded in the 2nd dimension. Defaults to
        None.
      training: (Optional) bool scalar, True if in training mode. Defaults to
        False.

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden], the input
        content stream in new representation, if `two_stream` is False; Or, a
        tuple of two float tensors of shape [batch_size, num_predict,
        hidden_size] and [batch_size, stack_size, m_seq_len, hidden_size], the
        output query stream and updated memory, if `two_stream` is True.
    """
    new_mems = []
    mems = [None] * self._stack_size if mems is None else tf.unstack(
        mems, axis=1)

    for i in range(self._stack_size):
      new_mems.append(cache_memory(
          content_stream, mems[i], self._mem_len, self._reuse_len))
      if self._tie_biases:
        content_bias, position_bias, segment_bias = (self._content_bias,
                                                     self._position_bias,
                                                     self._segment_bias)
      else:
        content_bias, position_bias, segment_bias = (self._content_bias[i],
                                                     self._position_bias[i],
                                                     self._segment_bias[i])

      outputs = self._stack[i](
          content_stream,
          content_mask,
          content_stream if mems[i] is None else tf.concat(
            [mems[i], content_stream], 1),
          position_encoding,
          content_bias,
          position_bias,
          self._segment_encoding[i],
          segment_matrix,
          segment_bias,
          query_stream=query_stream,
          query_mask=query_mask,
          target_mapping=target_mapping,
          training=training)

      if query_stream is None:
        content_stream = outputs
      else:
        content_stream, query_stream = outputs

    if query_stream is None:
      return content_stream
    else:
      new_mems = tf.stack(new_mems, axis=1)
      return query_stream, new_mems


class XLNetModel(tf.keras.layers.Layer):
  """XLNet model as described in https://arxiv.org/abs/1906.08237"""
  def __init__(self,
               vocab_size,
               mem_len=0,
               reuse_len=0,
               stack_size=6,
               hidden_size=512,
               num_heads=8,
               filter_size=2048,
               dropout_rate=0.0,
               dropout_rate_attention=0.0,
               tie_biases=False,
               two_stream=True,
               uni_data=False,
               filter_activation=tf.nn.relu):
    """Constructor.

    Args:
      vocab_size: int scalar, vocabulary size.
      mem_len: (Optional) int scalar, num tokens to be cached. Defaults to 0.
      reuse_len: (Optional) int scalar, num of tokens to be reused in the next
        batch. Defaults to 0.
      stack_size: (Optional) int scalar, num of layers in the decoder stack.
        Defaults to 6.
      hidden_size: (Optional) int scalar, the hidden size of continuous
        representation. Defaults to 512.
      num_heads: (Optional) int scalar, num of attention heads. Defaults to 8.
      filter_size: (Optional) int scalar, the depth of the intermediate dense
        layer of the feed-forward sublayer. Defaults to 2048.
      dropout_rate: (Optional) float scalar, dropout rate for Dropout layers.
        Defaults to 0.
      dropout_rate_attention: (Optional) float scalar, dropout rate applied on
        the query-to-reference attention matrix. Defaults to 0.
      tie_biases: bool scalar, whether to force all layers use the same content,
        position and segment bias (True), or create the biases for each layer (
        False).
      two_stream: (Optional) bool scalar, whether to process both content and
        query stream (True) or just content stream (False). Defaults to True.
      uni_data: (Optional) bool scalar, whether the data is unidirectional or
        bidirectional. Defaults to False. Defaults to False.
      filter_activation: (Optional) callable or string, activation function of
        the filter dense layer. Defaults to ReLU.
    """
    super(XLNetModel, self).__init__()
    self._vocab_size = vocab_size
    self._mem_len = mem_len
    self._reuse_len = reuse_len
    self._stack_size = stack_size
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate
    self._dropout_rate_attention = dropout_rate_attention
    self._tie_biases = tie_biases
    self._two_stream = two_stream
    self._uni_data = uni_data
    self._filter_activation = filter_activation

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
        tie_biases=tie_biases,
        filter_activation=filter_activation)

  def build(self, inputs_shape):
    """Creates weights of this layer.

    Args:
      inputs_shape: tuple of ints or 1-D int tensor. Not used.
    """
    if self._two_stream:
      self._mask_embedding = self.add_weight(
              'mask_embedding',
              shape=[self._hidden_size],
              initializer=tf.keras.initializers.RandomNormal(
                  mean=0., stddev=self._hidden_size ** -0.5),
              dtype='float32',
              trainable=True)
    super(XLNetModel, self).build(inputs_shape)

  def call(self,
           inputs,
           seg_ids,
           input_mask,
           target_mapping=None,
           mems=None,
           training=False):
    """Computes the output query stream and update the memories.

    Args:
      inputs: int tensor of shape [batch_size, q_seq_len], sequences of token
        IDs.
      seg_ids: int tensor of shape [batch_size, q_seq_len], segment ids where
        `seg_ids[b]` is a vector of segment IDs for each token in `inputs`.
      input_mask: float tensor of shape [batch_size, q_seq_len, q_seq_len],
        input mask where the `i`th token cannot attend the `j`th token if
        `input_mask[b, i, j] = 1`.
      target_mapping: (Optional) float tensor of shape [batch_size, num_predict,
        q_seq_len], where `target_mapping[b, i]` is the one-hot encoding of the
        index of the prediction target for the `i` prediction task (out of
        `num_predict`). May be zero-padded in the 2nd dimension. Defaults to
        None.
      mems: (Optional) float tensor of shape [batch_size, stack_size, m_seq_len
        , hidden_size], encodings of the memory sequences from the previous
        block. Defaults to None.
      training: (Optional) bool scalar, True if in training mode. Defaults to
        False.

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden], the input
        content stream in new representation, if `two_stream` is False; Or, a
        tuple of two float tensors of shape [batch_size, num_predict,
        hidden_size] and [batch_size, stack_size, m_seq_len, hidden_size], the
        output query stream and updated memory, if `two_stream` is True.
    """
    batch_size = tf.shape(inputs)[0]
    q_seq_len = tf.shape(inputs)[1]
    m_seq_len = 0 if mems is None else tf.shape(mems)[2]
    hidden_size = self._hidden_size
    content_mask, query_mask = compute_attention_mask(
        input_mask, m_seq_len, q_seq_len)

    relative_position_encoding = self._dropout_layer(
        get_position_encoding_xlnet(
            hidden_size, batch_size, m_seq_len, q_seq_len, self._uni_data),
        training=training)
    segment_matrix = compute_segment_matrix(
        seg_ids, m_seq_len, self._two_stream)

    content_stream = self._dropout_layer(
        self._embedding_layer(inputs), training=training)

    if self._two_stream:
      query_stream = self._dropout_layer(tf.tile(
          self._mask_embedding[tf.newaxis, tf.newaxis],
          [batch_size, tf.shape(target_mapping)[1], 1]), training=training)
    else:
      query_stream = None

    outputs = self._transformer_xl(
        content_stream,
        content_mask,
        relative_position_encoding,
        segment_matrix,
        query_stream,
        query_mask,
        mems,
        target_mapping, training)

    return outputs


class QuestionAnswerLogits(tf.keras.layers.Layer):
  """Computes prediction logits for question answering tasks."""
  def __init__(self, hidden_size, start_n_top, end_n_top, dropout_rate=0.0):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      start_n_top: int scalar, the number of top-scoring predictions for start
        position.
      end_n_top: int scalar, the number of top-scoring predictions for end
        position.
      dropout_rate: (Optional) float scalar, dropout rate for Dropout layers.
        Defaults to 0.
    """
    super(QuestionAnswerLogits, self).__init__()
    self._hidden_size = hidden_size
    self._start_n_top = start_n_top
    self._end_n_top = end_n_top
    self._dropout_rate = dropout_rate

    self._start_logits_dense_layer = tf.keras.layers.Dense(
        units=1, kernel_initializer=None)
    self._end_logits_dense_layer0 = tf.keras.layers.Dense(
        units=hidden_size,
        kernel_initializer=None,
        activation=tf.nn.tanh)
    self._end_logits_dense_layer1 = tf.keras.layers.Dense(
        units=1, kernel_initializer=None)
    self._end_logits_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12)
    self._answer_class_dense_layer0 = tf.keras.layers.Dense(
        units=hidden_size,
        kernel_initializer=None,
        activation=tf.nn.tanh)
    self._answer_class_dense_layer1 = tf.keras.layers.Dense(
        units=1,
        kernel_initializer=None,
        use_bias=False)
    self._answer_feature_dropout = tf.keras.layers.Dropout(
        rate=self._dropout_rate)

  def call(self,
           inputs,
           para_mask,
           cls_index,
           start_positions=None,
           training=True):
    """Computes logits for start position, end position and the `CLS` token.

    Args:
      inputs: float tensor of shape [batch_size, seq_len, hidden_size], vector
        representation for sequences.
      para_mask: float tensor of shape [batch_size, seq_len], mask for paragraph
        tokens.
      cls_index: int tensor of shape [batch_size], indices of the CLS token in
        `inputs`.
      start_positions: (Optional) int tensor of shape [batch_size], answer start
        positions. Defaults to None.
      training: (Optional) bool scalar, True if in training mode. Defaults to
        True.

    Returns:
      if training is True
        start_logits_masked: float tensor of shape [batch_size, seq_len], logits
          of predictions of start indices over the entire sequence.
        end_logits_masked: float tensor of shape [batch_size, seq_len], logits
          of predictions of end indices over the entire sequence.
        cls_logits: float tensor of shape [batch_size], logits of answerability
          predictions.
      if training is False
        start_top_log_probs: float tensor of shape [batch_size, start_n_top],
          logits of top scoring predictions of start indices.
        start_top_index: float tensor of shape [batch_size, start_n_top],
          start indices of top scoring predictions.
        end_top_log_probs: float tensor of shape [batch_size, start_n_top *
          end_n_top], logits of top scoring predictions of end indices.
        end_top_index: float tensor of shape [batch_size, start_n_top *
          end_n_top], end indices of top scoring predictions.
        cls_logits:  float tensor of shape [batch_size], logits of
          answerability predictions.
    """
    seq_len = tf.shape(inputs)[1]
    inputs = tf.transpose(inputs, [1, 0, 2])
    start_logits = self._start_logits_dense_layer(inputs)
    start_logits = tf.transpose(tf.squeeze(start_logits, -1), [1, 0])
    start_logits_masked = start_logits * (1 - para_mask
        ) + NEG_INF * para_mask
    if training:
      start_index = tf.one_hot(
          start_positions, depth=seq_len, axis=-1, dtype='float32')
      start_features = tf.einsum('TND,NT->ND', inputs, start_index)
      start_features = tf.tile(start_features[tf.newaxis], [seq_len, 1, 1])
      end_logits = self._end_logits_dense_layer0(
          tf.concat([inputs, start_features], axis=-1))
      end_logits = self._end_logits_layer_norm(end_logits)
      end_logits = self._end_logits_dense_layer1(end_logits)
      end_logits = tf.transpose(tf.squeeze(end_logits, -1), [1, 0])
      end_logits_masked = end_logits * (1 - para_mask
          ) + NEG_INF * para_mask
    else:
      start_log_probs = tf.nn.log_softmax(start_logits_masked, -1)
      start_top_log_probs, start_top_index = tf.nn.top_k(
          start_log_probs, k=self._start_n_top)
      start_index = tf.one_hot(
          start_top_index, depth=seq_len, axis=-1, dtype='float32')
      start_features = tf.einsum('TND,NKT->NKD', inputs, start_index)
      end_input = tf.tile(inputs[:, :, tf.newaxis],
                          [1, 1, self._start_n_top, 1])
      start_features = tf.tile(start_features[tf.newaxis], [seq_len, 1, 1, 1])
      end_input = tf.concat([end_input, start_features], axis=-1)
      end_logits = self._end_logits_dense_layer0(end_input)
      end_logits = tf.reshape(end_logits, [seq_len, -1, self._hidden_size])
      end_logits = self._end_logits_layer_norm(end_logits)

      end_logits = tf.reshape(end_logits,
          [seq_len, -1, self._start_n_top, self._hidden_size])

      end_logits = self._end_logits_dense_layer1(end_logits)
      end_logits = tf.reshape(end_logits, [seq_len, -1, self._start_n_top])
      end_logits = tf.transpose(end_logits, [1, 2, 0])
      end_logits_masked = end_logits * (1 - para_mask[:, tf.newaxis]
          ) + NEG_INF * para_mask[:, tf.newaxis]
      end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
      end_top_log_probs, end_top_index = tf.nn.top_k(
          end_log_probs, k=self._end_n_top)
      end_top_log_probs = tf.reshape(end_top_log_probs,
                                     [-1, self._start_n_top * self._end_n_top])
      end_top_index = tf.reshape(end_top_index,
                                 [-1, self._start_n_top * self._end_n_top])

    cls_index = tf.one_hot(cls_index, seq_len, axis=-1, dtype='float32')
    cls_feature = tf.einsum('TND,NT->ND', inputs, cls_index)
    start_p = tf.nn.softmax(start_logits_masked, axis=-1)
    start_feature = tf.einsum('TND,NT->ND', inputs, start_p)
    ans_feature = tf.concat([start_feature, cls_feature], -1)
    ans_feature = self._answer_class_dense_layer0(ans_feature)
    ans_feature = self._answer_feature_dropout(ans_feature, training=training)
    cls_logits = self._answer_class_dense_layer1(ans_feature)
    cls_logits = tf.squeeze(cls_logits, -1)

    if training:
      return start_logits_masked, end_logits_masked, cls_logits
    else:
      return (start_top_log_probs,
              start_top_index,
              end_top_log_probs,
              end_top_index,
              cls_logits)


class PretrainingXLNet(XLNetModel):
  """XLNet purposed for pretraining."""
  def __init__(self,
               vocab_size,
               mem_len,
               reuse_len,
               stack_size=6,
               hidden_size=512,
               num_heads=8,
               filter_size=2048,
               dropout_rate=0.0,
               dropout_rate_attention=0.0,
               tie_biases=False):
    """Constructor.

    Args:
      vocab_size: int scalar, vocabulary size.
      mem_len: int scalar, num tokens to be cached.
      reuse_len: int scalar, num of tokens to be reused in the next batch.
      stack_size: (Optional) int scalar, num of layers in the decoder stack.
        Defaults to 6.
      hidden_size: (Optional) int scalar, the hidden size of continuous
        representation. Defaults to 512.
      num_heads: (Optional) int scalar, num of attention heads. Defaults to 8.
      filter_size: (Optional) int scalar, the depth of the intermediate dense
        layer of the feed-forward sublayer. Defaults to 2048.
      dropout_rate: (Optional) float scalar, dropout rate for Dropout layers.
        Defaults to 0.
      dropout_rate_attention: (Optional) float scalar, dropout rate applied on
        the query-to-reference attention matrix. Defaults to 0.
      tie_biases: (Optional) bool scalar, whether to force all layers use the
        same content, position and segment bias (True), or create the biases for
        each layer (False). Defaults to False.
    """
    super(PretrainingXLNet, self).__init__(
        vocab_size=vocab_size,
        mem_len=mem_len,
        reuse_len=reuse_len,
        stack_size=stack_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        filter_size=filter_size,
        dropout_rate=dropout_rate,
        dropout_rate_attention=dropout_rate_attention,
        tie_biases=tie_biases,
        two_stream=True,
        uni_data=False,
        filter_activation=tf.nn.relu)
    self._dense_layer_output = tf.keras.layers.Dense(
          units=hidden_size,
          kernel_initializer=None,
          activation=lambda x: tf.keras.activations.gelu(x, approximate=True))
    self._layernorm_output = tf.keras.layers.LayerNormalization(epsilon=1e-12)

  def build(self, inputs_shape):
    """Creates weights of this layer.

    Args:
      inputs_shape: tuple of ints or 1-D int tensor. Not used.
    """
    self._bias_output= self.add_weight(
        'bias_output',
        shape=[self._vocab_size],
        initializer=tf.zeros_initializer(),
        dtype='float32',
        trainable=True)
    super(PretrainingXLNet, self).build(inputs_shape)

  def call(self, inputs, seg_ids, perm_mask, target_mapping, mems):
    """Compute permutation language modeling logits and update memory.

    Args:
      inputs: int tensor of shape [batch_size, q_seq_len], sequences of token
        IDs.
      seg_ids: int tensor of shape [batch_size, q_seq_len], segment ids where
        `seg_ids[b]` is a vector of segment IDs for each token in `inputs`.
      perm_mask: float tensor of shape [batch_size, q_seq_len, q_seq_len],
        permutation mask where the `i`th token cannot attend the
        `j`th token if `perm_mask[b, i, j] = 1`.
      target_mapping: float tensor of shape [batch_size, num_predict,
        q_seq_len], where `target_mapping[b, i]` is the one-hot encoding of
        the index of the prediction target for the `i` prediction task (out of
        `num_predict`). May be zero-padded in the 2nd dimension.
      mems: float tensor of shape [batch_size, stack_size, m_seq_len
        , hidden_size], encodings of the memory sequences from the previous
        block.

    Returns:
      logits: float tensor of shape [batch_size, num_predict, vocab_size],
        logits of predicted tokens over the vocabulary.
      new_mems: float tensor of shape [batch_size, stack_size, m_seq_len,
        hidden_size], the updated memory.
    """
    outputs, new_mems = super(PretrainingXLNet, self).call(inputs,
                                                           seg_ids,
                                                           perm_mask,
                                                           target_mapping,
                                                           mems,
                                                           training=True)
    outputs = self._layernorm_output(self._dense_layer_output(outputs))
    logits = tf.einsum('NPD,VD->NPV', outputs, self._embedding_layer.weights[0]
        ) + self._bias_output

    return logits, new_mems


class QuestionAnswerXLNet(XLNetModel):
  """XLNet model for question-answer tasks (e.g. SQuAD).

  Computes logits for start position, end position and the answerability.
  """
  def __init__(self,
               vocab_size,
               stack_size=6,
               hidden_size=512,
               num_heads=8,
               filter_size=2048,
               dropout_rate=0.0,
               dropout_rate_attention=0.0,
               tie_biases=False,
               start_n_top=5,
               end_n_top=5):
    """Constructor.

    Args:
      vocab_size: int scalar, vocabulary size.
      stack_size: (Optional) int scalar, num of layers in the decoder stack.
        Defaults to 6.
      hidden_size: (Optional) int scalar, the hidden size of continuous
        representation. Defaults to 512.
      num_heads: (Optional) int scalar, num of attention heads. Defaults to 8.
      filter_size: (Optional) int scalar, the depth of the intermediate dense
        layer of the feed-forward sublayer. Defaults to 2048.
      dropout_rate: (Optional) float scalar, dropout rate for Dropout layers.
        Defaults to 0.
      dropout_rate_attention: (Optional) float scalar, dropout rate applied on
        the query-to-reference attention matrix. Defaults to 0.
      tie_biases: (Optional) bool scalar, whether to force all layers use the
        same content, position and segment bias (True), or create the biases for
        each layer (False). Defaults to False.
      start_n_top: (Optional) int scalar, the number of top-scoring predictions
        for start position. Defaults to 5.
      end_n_top: (Optional) int scalar, the number of top-scoring predictions
        for end position. Defaults to 5.
    """
    super(QuestionAnswerXLNet, self).__init__(
        vocab_size=vocab_size,
        mem_len=0,
        reuse_len=0,
        stack_size=stack_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        filter_size=filter_size,
        dropout_rate=dropout_rate,
        dropout_rate_attention=dropout_rate_attention,
        tie_biases=tie_biases,
        two_stream=False,
        uni_data=True,
        filter_activation=tf.nn.gelu)
    self._logits_layer = QuestionAnswerLogits(
        hidden_size, start_n_top, end_n_top, dropout_rate=dropout_rate)

  def call(self,
           inputs,
           seg_ids,
           input_mask,
           para_mask,
           cls_index,
           start_positions=None,
           training=False):
    """Computes logits for start position, end position, and answerability.

    Args:
      inputs: int tensor of shape [batch_size, q_seq_len], sequences of token
        IDs.
      seg_ids: int tensor of shape [batch_size, q_seq_len], segment ids where
        `seg_ids[b]` is a vector of segment IDs for each token in `inputs`.
      input_mask: float tensor of shape [batch_size, 1, q_seq_len], input mask
        mask where the `i`th token cannot attend the `j`th token if
        `input_mask[b, i, j] = 1`.
      para_mask: bool tensor of shape [batch_size, q_seq_len], paragraph mask
        where the `para_mask[b, i]` is 0 if the `i`th token is paragraph token.
      cls_index: int tensor of shape [batch_size], indices of the CLS token in
        `inputs`.
      start_positions: (Optional) int tensor of shape [batch_size], answer start
        positions. Defaults to None.
      training: (Optional) bool scalar, True if in training mode. Defaults to
        False.

    Returns:
      if training is True
        start_logits_masked: float tensor of shape [batch_size, seq_len], logits
          of predictions of start indices over the entire sequence.
        end_logits_masked: float tensor of shape [batch_size, seq_len], logits
          of predictions of end indices over the entire sequence.
        cls_logits: float tensor of shape [batch_size], logits of answerability
          predictions.
      if training is False
        start_top_log_probs: float tensor of shape [batch_size, start_n_top],
          logits of top scoring predictions of start indices.
        start_top_index: float tensor of shape [batch_size, start_n_top],
          start indices of top scoring predictions.
        end_top_log_probs: float tensor of shape [batch_size, start_n_top *
          end_n_top], logits of top scoring predictions of end indices.
        end_top_index: float tensor of shape [batch_size, start_n_top *
          end_n_top], end indices of top scoring predictions.
        cls_logits:  float tensor of shape [batch_size], logits of
          answerability predictions.
    """
    outputs = super(QuestionAnswerXLNet, self).call(
        inputs, seg_ids, input_mask, training=training)
    outputs = self._logits_layer(outputs,
                                 para_mask,
                                 cls_index,
                                 start_positions=start_positions,
                                 training=training)
    return outputs


class ClassificationXLNet(XLNetModel):
  """XLNet for sequence (or sequence pair) classification."""
  def __init__(self,
               vocab_size,
               stack_size=6,
               hidden_size=512,
               num_heads=8,
               filter_size=2048,
               dropout_rate=0.0,
               dropout_rate_attention=0.0,
               tie_biases=False,
               num_classes=2):
    """Constructor.

    Args:
      vocab_size: int scalar, vocabulary size.
      stack_size: (Optional) int scalar, num of layers in the decoder stack.
        Defaults to 6.
      hidden_size: (Optional) int scalar, the hidden size of continuous
        representation. Defaults to 512.
      num_heads: (Optional) int scalar, num of attention heads. Defaults to 8.
      filter_size: (Optional) int scalar, the depth of the intermediate dense
        layer of the feed-forward sublayer. Defaults to 2048.
      dropout_rate: (Optional) float scalar, dropout rate for Dropout layers.
        Defaults to 0.
      dropout_rate_attention: (Optional) float scalar, dropout rate applied on
        the query-to-reference attention matrix. Defaults to 0.
      tie_biases: (Optional) bool scalar, whether to force all layers use the
        same content, position and segment bias (True), or create the biases for
        each layer (False). Defaults to False.
      num_classes: (Optional) int scalar, num of classes. Defaults to 2.
    """
    super(ClassificationXLNet, self).__init__(
        vocab_size=vocab_size,
        mem_len=0,
        reuse_len=0,
        stack_size=stack_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        filter_size=filter_size,
        dropout_rate=dropout_rate,
        dropout_rate_attention=dropout_rate_attention,
        tie_biases=tie_biases,
        two_stream=False,
        uni_data=True,
        filter_activation=tf.nn.gelu)
    self._dense_layer_output = tf.keras.layers.Dense(
        units=hidden_size, kernel_initializer=None, activation=tf.nn.tanh)
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    self._dense_layer_logits = tf.keras.layers.Dense(
        units=num_classes, kernel_initializer=None)

  def call(self, inputs, seg_ids, input_mask, training=False):
    """Computes logits for sequence label predictions.

    Args:
      inputs: float tensor of shape [batch_size, seq_len], sequences of token
        IDs.
      seg_ids: int tensor of shape [batch_size, seq_len], segment ids where
        `seg_ids[b]` is a vector of segment IDs for each token in `inputs`.
      input_mask: float tensor of shape [batch_size, 1, seq_len], input mask
        where `input_mask[b, :, i] = 1` if the `i`th token is to be masked.
      training: (Optional) bool scalar, True if in training mode. Defaults to
        False.

    Returns:
      logits: float tensor of shape [batch_size, num_classes], logits of
        sequence label predictions.
    """
    outputs = super(ClassificationXLNet, self).call(
        inputs, seg_ids, input_mask, training=training)[:, -1]
    outputs = self._dropout_layer(
        self._dense_layer_output(outputs), training=training)
    logits = self._dense_layer_logits(outputs)
    return logits

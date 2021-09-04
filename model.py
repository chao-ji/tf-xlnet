"""Defines XLNet model in tf.keras.API."""
import tensorflow as tf

from commons.layers import FeedForwardNetwork
from commons.layers import RelativeAttention

from utils import compute_attention_mask
from utils import compute_position_encoding 
from utils import compute_segment_matrix 
from utils import cache_memory 
from utils import get_position_encoding


class DecoderLayer(tf.keras.layers.Layer):
  """The building block that makes the decoder stack of layers, consisting of a 
  self-attention sublayer and a feed-forward sublayer. Takes content stream (and
  optionally query stream) as input sequences.
  """
  def __init__(self,
               hidden_size,
               num_heads,
               filter_size,
               dropout_rate,
               dropout_rate_attention,
               filter_activation=tf.nn.relu):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      filter_size: int scalar, the depth of the intermediate dense layer of the
        feed-forward sublayer.
      dropout_rate: float scalar, dropout rate for the Dropout layers.
      dropout_rate_attention: float scalar, dropout rate applied on the
        query-to-reference attention matrix.
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
      query_stream: (Optional) float tensor of shape [batch_size,
        num_predictions, hidden_size], the query stream for pre-training XLNet. 
      query_mask: (Optional) float tensor of shape [batch_size, 1, q_seq_len,
        c_seq_len], token mask for query stream.
      target_mapping: (Optional) float tensor of shape [batch_size,
        num_predictions, hidden_size], one-hot encodings of the indices of
        prediction targets. 
      training: (Optional) bool scalar, True if in training mode.

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden_size], for
        single stream input; or a tuple of two tensors of shape [batch_size,
        q_seq_len, hidden_size] and [batch_size, num_targets, hidden_size].
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
  """TransformerXL adapted to optionally process query stream in addition to
  content stram.
  """
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
               tie_biases=True,
               filter_activation=tf.nn.relu):
    """Constructor.

    Args:
      vocab_size: int scalar, vocabulary size.
      stack_size: (Optional) int scalar, num of layers in the decoder stack.
      hidden_size: (Optional) int scalar, the hidden size of continuous
        representation.
      num_heads: (Optional) int scalar, num of attention heads.
      filter_size: (Optional) int scalar, the depth of the intermediate dense
        layer of the feed-forward sublayer.
      mem_len: (Optional) int scalar, num tokens to be cached.
      reuse_len: (Optional) int scalar, num of tokens to be reused in the next
        batch. 
      dropout_rate: (Optional) float scalar, dropout rate for Dropout layers.
      dropout_rate_attention: (Optional) float scalar, dropout rate applied on
        the query-to-reference attention matrix. 
      tie_biases: (Optional) bool scalar, whether to force all layers use the
        same content bias and position bias (True), or create the biases for
        each layer (False).
      two_steram: (Optional) bool scalar, whether to apply multi-headed
        attention with both content and query stream (True), or just content
        stream (False).
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
      query_stream: (Optional) float tensor of shape [batch_size, num_targets,
        hidden_size], input query stream.
      query_mask: (Optional) float tensor of shape [batch_size, 1, q_seq_len,
        m_seq_len + q_seq_len], permutation mask for the query stream.
      mems: (Optional) float tensor of shape [batch_size, stack_size, m_seq_len
        , hidden_size], encodings of the memory sequences from the previous
        block.
      target_mapping: (Optional) float tensor of shape [batch_size, num_targets,
        q_seq_len], one-hot encodings of the indices of prediction targets.
      training: (Optional) bool scalar, True if in training mode.
 
    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden], the input
        query stream in new representation, if `two_stream` is False; Or, a 
        tuple of two float tensors of shape [batch_size, num_targets,
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
      segment_encoding = self._segment_encoding

      outputs = self._stack[i](
          content_stream,
          content_mask,
          content_stream if mems[i] is None else tf.concat(
            [mems[i], content_stream], 1),
          position_encoding,
          content_bias,
          position_bias,
          segment_encoding[i],
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
  """XLNet model for pretraining as described in 
  https://arxiv.org/abs/1906.08237
  """
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
               tie_biases=False,
               two_stream=True,
               uni_data=False,
               filter_activation=tf.nn.relu):
    """Constructor.

    Args:
      vocab_size: (Optional) int scalar, vocabulary size.
      stack_size: (Optional) int scalar, num of layers in the decoder stack.
      hidden_size: (Optional) int scalar, the hidden size of continuous
        representation.
      num_heads: (Optional) int scalar, num of attention heads.
      filter_size: (Optional) int scalar, the depth of the intermediate dense
        layer of the feed-forward sublayer.
      mem_len: (Optional) int scalar, num tokens to be cacched.
      reuse_len: (Optional) int scalar, num of tokens to be reused in the next
        batch.
      dropout_rate: (Optional) float scalar, dropout rate for Dropout layers.
      dropout_rate_attention: (Optional) float scalar, dropout rate applied on
        the query-to-reference attention matrix. 
      tie_biases: bool scalar, whether to force all layers use the same
        content, position and segment bias (True), or create the biases for each
        layer (False).
      two_stream: (Optional) bool scalar, whether to process both content and
        query stream (True) or just content stream (False).
      uni_data: (Optional) bool scalar, whether the data is unidirectional or
        bidirectional. Defaults to False.
      filter_activation: (Optional) callable or string, activation function of
        the filter dense layer. Defaults to ReLU.
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
              shape=[1, 1, self._hidden_size],
              initializer=tf.keras.initializers.RandomNormal(
                  mean=0., stddev=self._hidden_size ** -0.5),
              dtype='float32',
              trainable=True)
    super(XLNetModel, self).build(inputs_shape)

  def call(self, 
           inputs,
           segment_ids,
           perm_mask,
           target_mapping=None,
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
      target_mapping: (Optional) float tensor of shape [batch_size, num_targets,
        q_seq_len], one-hot encodings of the indices of prediction targets.
      mems: (Optional) float tensor of shape [batch_size, stack_size, m_seq_len
        , hidden_size], encodings of the memory sequences from the previous
        block.

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden], the input
        query stream in new representation, if `two_stream` is False; Or, a 
        tuple of two float tensors of shape [batch_size, num_targets,
        hidden_size] and [batch_size, stack_size, m_seq_len, hidden_size], the
        output query stream and updated memory, if `two_stream` is True.
    """
    batch_size = tf.shape(inputs)[0]
    q_seq_len = tf.shape(inputs)[1]
    m_seq_len = 0 if mems is None else tf.shape(mems[0])[1]
    content_mask, query_mask = compute_attention_mask(
        perm_mask, m_seq_len, q_seq_len)

    relative_position_encoding = self._dropout_layer(compute_position_encoding(
        self._hidden_size, batch_size, m_seq_len, q_seq_len, self._uni_data))
    segment_matrix = compute_segment_matrix(
        segment_ids, m_seq_len, self._two_stream)

    content_stream = self._dropout_layer(self._embedding_layer(inputs))

    if self._two_stream:
      query_stream = self._dropout_layer(tf.tile(self._mask_embedding,
          [batch_size, tf.shape(target_mapping)[1], 1]))
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
        target_mapping)

    return outputs


class PretrainLogits(tf.keras.layers.Layer):
  """Converts query stream to final prediction logit for the permutation
  language model objects.
  """
  def __init__(self, hidden_size, vocab_size):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      vocab_size: int scalar, vocabulary size.
    """
    super(PretrainLogits, self).__init__()
    self._hidden_size = hidden_size
    self._vocab_size = vocab_size

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
    self._bias_output= self.add_weight(
        'bias_output',
        shape=[self._vocab_size],
        initializer=tf.zeros_initializer())
    super(PretrainLogits, self).build(inputs_shape)

  def call(self, query_stream, embeddings):
    """Computes logits tensor.

    Args:
      query_stream: float tensor of shape [batch_size, num_targets,
        hidden_size], input query stream.
      embeddings: float tensor of shape [vocab_size, hidden_size], embedding
        vectors for all tokens in the vocabulary.

    Returns:
      logits: float tensor of shape [batch_size, num_targets, vocab_size],
        logits over vocabulary.
    """
    outputs = self._layernorm_output(self._dense_output(query_stream))
    logits = tf.einsum('NPD,VD->NPV', outputs, embeddings 
        ) + self._bias_output
    return logits


class QuestionAnwserLogits(tf.keras.layers.Layer):
  """Computes prediction logits for question answering tasks."""
  def __init__(self, hidden_size, start_n_top, end_n_top, dropout_rate=0.1):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      start_n_top: int scalar, the number of top-scoring predictions for start
        position.
      end_n_top: int scalar, the number of top-scoring predictions for end
        postion.
      dropout_rate: (Optional) float scalar, dropout rate for Dropout layers.
    """
    super(QuestionAnwserLogits, self).__init__()
    self._hidden_size = hidden_size
    self._start_n_top = start_n_top
    self._end_n_top = end_n_top
    self._dropout_rate = dropout_rate

    self.start_logits_proj_layer = tf.keras.layers.Dense(
        units=1, kernel_initializer=None)
    self.end_logits_proj_layer0 = tf.keras.layers.Dense(
        units=hidden_size,
        kernel_initializer=None,
        activation=tf.nn.tanh)
    self.end_logits_proj_layer1 = tf.keras.layers.Dense(
        units=1, kernel_initializer=None)
    self.end_logits_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12)
    self.answer_class_proj_layer0 = tf.keras.layers.Dense(
        units=hidden_size,
        kernel_initializer=None,
        activation=tf.nn.tanh)
    self.answer_class_proj_layer1 = tf.keras.layers.Dense(
        units=1,
        kernel_initializer=None,
        use_bias=False)
    self.ans_feature_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)

  def call(self,
           inputs,
           paragraph_mask,
           cls_index,
           start_positions=None,
           end_positions=None,
           is_impossible=None,
           training=True):
    """Computes logits for start position, end position and the `CLS` token.

    Args:
      inputs: float tensor of shape [batch_size, seq_len, hidden_size], vector
        representation for sequences.
      paragraph_mask: float tensor of shape [batch_size, seq_len], mask for
        paragraph tokens.
      cls_index: int tensor of shape [batch_size], indices of the CLS token in
        `inputs`.
      start_positions: (Optional )int tensor of shape [batch_size], answer start
        positions.
      end_positions: (Optional) int tensor of shape [batch_size], answer end
        positions.
      is_impossible: (Optional) float tensor of shape [batch_size], indicating
        if question is answerable.
      training: (Optional) bool scalar, True if in training mode.
    """
    seq_len = tf.shape(inputs)[1]  
    inputs = tf.transpose(inputs, [1, 0, 2])
    start_logits = self.start_logits_proj_layer(inputs)
    start_logits = tf.transpose(tf.squeeze(start_logits, -1), [1, 0])
    start_logits_masked = start_logits * (1 - paragraph_mask
        ) - 1e30 * paragraph_mask
    start_log_probs = tf.nn.log_softmax(start_logits_masked, -1)
    if training:
      start_positions = tf.reshape(start_positions, [-1])
      start_index = tf.one_hot(
          start_positions, depth=seq_len, axis=-1, dtype='float32')
      start_features = tf.einsum("TND,NT->ND", inputs, start_index)
      start_features = tf.tile(start_features[tf.newaxis], [seq_len, 1, 1])
      end_logits = self.end_logits_proj_layer0(
          tf.concat([inputs, start_features], axis=-1))
      end_logits = self.end_logits_layer_norm(end_logits)
      end_logits = self.end_logits_proj_layer1(end_logits)
      end_logits = tf.transpose(tf.squeeze(end_logits, -1), [1, 0])
      end_logits_masked = end_logits * (1 - paragraph_mask
          ) - 1e30 * paragraph_mask
      end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
    else:
      start_top_log_probs, start_top_index = tf.nn.top_k(
          start_log_probs, k=self._start_n_top)
      start_index = tf.one_hot(
          start_top_index, depth=seq_len, axis=-1, dtype='float32')
      start_features = tf.einsum("TND,NKT->NKD", inputs, start_index)
      end_input = tf.tile(inputs[:, :, tf.newaxis], [1, 1, self._start_n_top, 1])
      start_features = tf.tile(start_features[tf.newaxis], [seq_len, 1, 1, 1])
      end_input = tf.concat([end_input, start_features], axis=-1)
      end_logits = self.end_logits_proj_layer0(end_input)
      end_logits = tf.reshape(end_logits, [seq_len, -1, self._hidden_size])
      end_logits = self.end_logits_layer_norm(end_logits)

      end_logits = tf.reshape(end_logits,
          [seq_len, -1, self._start_n_top, self._hidden_size])

      end_logits = self.end_logits_proj_layer1(end_logits)
      end_logits = tf.reshape(end_logits, [seq_len, -1, self._start_n_top])
      end_logits = tf.transpose(end_logits, [1, 2, 0])
      end_logits_masked = end_logits * (
          1 - paragraph_mask[:, tf.newaxis]) - 1e30 * paragraph_mask[:, tf.newaxis]
      end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
      end_top_log_probs, end_top_index = tf.nn.top_k(
          end_log_probs, k=self._end_n_top)
      end_top_log_probs = tf.reshape(end_top_log_probs,
                                     [-1, self._start_n_top * self._end_n_top])
      end_top_index = tf.reshape(end_top_index,
                                 [-1, self._start_n_top * self._end_n_top])

    if training:
      outputs = {"start_log_probs": start_log_probs,
                 "end_log_probs": end_log_probs}
    else:
      outputs = {"start_top_log_probs": start_top_log_probs,
                 "start_top_index": start_top_index,
                 "end_top_log_probs": end_top_log_probs,
                 "end_top_index": end_top_index}

    cls_index = tf.one_hot(cls_index, seq_len, axis=-1, dtype='float32')
    cls_feature = tf.einsum("TND,NT->ND", inputs, cls_index)
    start_p = tf.nn.softmax(start_logits_masked, axis=-1)
    start_feature = tf.einsum("TND,NT->ND", inputs, start_p)
    ans_feature = tf.concat([start_feature, cls_feature], -1)
    ans_feature = self.answer_class_proj_layer0(ans_feature)
    ans_feature = self.ans_feature_dropout(ans_feature)
    cls_logits = self.answer_class_proj_layer1(ans_feature)
    cls_logits = tf.squeeze(cls_logits, -1)
    outputs["cls_logits"] = cls_logits

    if not training:
      return outputs

    return start_logits_masked, end_logits_masked, cls_logits


class PretrainingXLNet(XLNetModel):
  def __init__(self,
               vocab_size,
               stack_size,
               hidden_size,
               num_heads,
               filter_size,
               mem_len,
               reuse_len,
               dropout_rate,
               dropout_rate_attention,
               tie_biases):
    super(PretrainingXLNet, self).__init__(
        vocab_size=vocab_size,
        stack_size=stack_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        filter_size=filter_size,
        mem_len=mem_len,
        reuse_len=reuse_len,
        dropout_rate=dropout_rate,
        dropout_rate_attention=dropout_rate_attention,
        tie_biases=tie_biases)
    self._logits_layer = PretrainLogits(hidden_size=hidden_size,
                                        vocab_size=vocab_size)

  def call(self, input_ids, seg_ids, perm_mask, target_mapping, mems):
    model_output, new_mems = super(PretrainingXLNet, self).call(
        input_ids, seg_ids, perm_mask, target_mapping, mems)
    logits = self._logits_layer(
        model_output, self._embedding_layer.weights[0])
    return logits, new_mems


class QuestionAnswerXLNet(XLNetModel):
  """XLNet model for question-answer tasks (e.g. SQuAD).

  Computes logits for start position, end position and the `CLS` token.
  """
  def __init__(self, 
               vocab_size,
               stack_size,
               hidden_size,
               num_heads,
               filter_size,
               mem_len,
               reuse_len,
               dropout_rate,
               dropout_rate_attention,
               tie_biases,
               two_stream,
               uni_data,
               filter_activation,
               start_n_top,
               end_n_top):
    super(QuestionAnswerXLNet, self).__init__(
        vocab_size=vocab_size,
        stack_size=stack_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        filter_size=filter_size,
        mem_len=mem_len,
        reuse_len=reuse_len,
        dropout_rate=dropout_rate,
        dropout_rate_attention=dropout_rate_attention,
        tie_biases=tie_biases,
        two_stream=two_stream,
        uni_data=uni_data,
        filter_activation=filter_activation)
    self._logits_layer = QuestionAnwserLogits(
        hidden_size, start_n_top, end_n_top, dropout_rate=0.)

  def call(self,
           input_ids,
           segment_ids,
           perm_mask,
           p_mask,
           cls_index,
           start_positions=None,
           end_positions=None,
           is_impossible=None,
           training=False):
    model_output = super(QuestionAnswerXLNet, self).call(
        input_ids, segment_ids, perm_mask)
    outputs = self._logits_layer(model_output,
                                 p_mask,
                                 cls_index,
                                 start_positions=start_positions,
                                 end_positions=end_positions,
                                 is_impossible=is_impossible,
                                 training=training)
    return outputs


class ClassificationXLNet(XLNetModel):
  def __init__(self,
               vocab_size,
               stack_size,
               hidden_size,
               num_heads,
               filter_size,
               mem_len,
               reuse_len,
               dropout_rate,
               dropout_rate_attention,
               tie_biases,
               two_stream,
               uni_data,
               filter_activation,
               num_classes):
    super(ClassificationXLNet, self).__init__(
        vocab_size=vocab_size,
        stack_size=stack_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        filter_size=filter_size,
        mem_len=mem_len,
        reuse_len=reuse_len,
        dropout_rate=dropout_rate,
        dropout_rate_attention=dropout_rate_attention,
        tie_biases=tie_biases,
        two_stream=two_stream,
        uni_data=uni_data,
        filter_activation=filter_activation)

    self._proj_layer = tf.keras.layers.Dense(
        units=hidden_size, kernel_initializer=None, activation=tf.nn.tanh)
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    self._proj_layer1 = tf.keras.layers.Dense(
        units=num_classes, kernel_initializer=None)

  def call(self, input_ids, segment_ids, input_mask):
    outputs = super(ClassificationXLNet, self).call(
        input_ids, segment_ids, input_mask)
 
    summary = outputs[:, -1, :]
    summary = self._proj_layer(summary)
    summary = self._dropout_layer(summary)

    logits = self._proj_layer1(summary)

    return logits

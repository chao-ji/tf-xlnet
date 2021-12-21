"""Defines functions to be used for building XLNet model architecture."""
import tensorflow as tf

from text_utils import SEG_ID_CLS


def cache_memory(inputs, memory=None, mem_len=0, reuse_len=0):
  """Cache the memory for the next segment.

  Args:
    inputs: float tensor of shape [batch_size, seq_len, hidden_size], input
      sequences.
    memory: (Optional) float tensor of shape [batch_size, m_seq_len, hidden_size
      ], memory for the current segment.
    mem_len: (Optional) int scalar, num tokens to be cached.
    reuse_len: (Optional) int scalar, length of the input sequences to be reused
      as part of the cache memory sequences.

  Returns:
    new_memory: float tensor of shape [batch_size, mem_len, hidden_size]
  """
  if reuse_len > 0:
    inputs = inputs[:, :reuse_len]
  if memory is None:
    new_memory = inputs[:, -mem_len:]
  else:
    new_memory = tf.concat([memory, inputs], 1)[:, -mem_len:]
  return tf.stop_gradient(new_memory)


def compute_attention_mask(perm_mask, m_seq_len, q_seq_len):
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


def get_position_encoding_xlnet(
    hidden_size, batch_size, m_seq_len, q_seq_len, uni_data=False):
  """Computes position encoding matrix for XLNet.

  Args:
    hidden_size: int scalar, the hidden size of continuous representation.
    batch_size: int scalar tensor, batch size.
    m_seq_len: int scalar tensor, memory sequence length.
    q_seq_len: int scalar tensor, query sequence length.
    uni_data: (Optional) bool scalar, whether the data is unidirectional or
      bidirectional. Defaults to False.

  Returns:
    relative_position_encoding: float tensor of shape [batch_size, m_seq_len +
      2 * q_seq_len, hidden_size], the tensor that encodes position
      information.
  """
  def uni_data_position_encoding(hidden_size, batch_size, pos_seq):
    position_encoding = get_position_encoding(pos_seq, hidden_size)
    position_encoding = tf.tile(position_encoding[tf.newaxis],
                                [batch_size, 1, 1])
    return position_encoding

  fw_pos_seq = tf.range(m_seq_len + q_seq_len, -q_seq_len, -1.0)

  if uni_data:
    fw_position_encoding = uni_data_position_encoding(
        hidden_size, batch_size, fw_pos_seq)
    return fw_position_encoding
  else:
    fw_position_encoding = uni_data_position_encoding(
        hidden_size, batch_size // 2, fw_pos_seq)

    bw_pos_seq = tf.range(-m_seq_len - q_seq_len, q_seq_len, 1.0)
    bw_position_encoding = uni_data_position_encoding(
        hidden_size, batch_size // 2, bw_pos_seq)

    relative_position_encoding = tf.concat(
      [fw_position_encoding, bw_position_encoding], axis=0)
    return relative_position_encoding


def compute_segment_matrix(segment_ids, m_seq_len, use_cls_mask=True):
  """Computes the binary matrix indicating whether two positions are from the
  same segment or not.

  Args:
    segment_ids: int tensor of shape [batch_size, q_seq_len], where
        `segment_ids[b]` is an vector of segment IDs for each token in
        `token_ids`.
    m_seq_len: int scalar, memory sequence length.
    use_cls_mask: (Optional) bool scalar, whether to use mask for the special
      token CLS. Defaults to True.

  Returns:
    segment_matrix: bool tensor of shape [batch_size, q_seq_len, m_seq_len +
      q_seq_len], binary matrix indicating whether two positions are from the
      same segment or not.
  """
  reference_segment_ids = tf.pad(segment_ids, [[0, 0], [m_seq_len, 0]])

  if use_cls_mask:
    class_index_matrix = tf.logical_or(
        (segment_ids == SEG_ID_CLS)[..., tf.newaxis],
        (reference_segment_ids == SEG_ID_CLS)[:, tf.newaxis])

    segment_matrix = (segment_ids[..., tf.newaxis] ==
                      reference_segment_ids[:, tf.newaxis])

    segment_matrix = tf.logical_or(class_index_matrix, segment_matrix)
  else:
    segment_matrix = tf.logical_not(
        tf.equal(segment_ids[..., tf.newaxis],
                 reference_segment_ids[:, tf.newaxis]))
  return segment_matrix


def get_position_encoding(pos_seq, hidden_size):
  """Creates a tensor that encodes positional information.

  Args:
    pos_seq: int tensor of shape [seq_len], the sequence of relative distances.
    hidden_size: int scalar, the hidden size of continuous representation.

  Returns:
    position_encoding: float tensor of shape [batch_size, seq_len, hidden_size],
      the tensor that encodes positional information.
  """
  inverse_frequencies = 1 / (10000 ** (tf.range(0, hidden_size, 2.0) /
      hidden_size))
  position_encoding = tf.einsum('i,j->ij', pos_seq, inverse_frequencies)
  position_encoding = tf.concat([tf.sin(position_encoding),
                                   tf.cos(position_encoding)], axis=1)
  return position_encoding

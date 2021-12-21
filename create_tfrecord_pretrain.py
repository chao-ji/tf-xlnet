"""Convert corpus in raw text into TFRecord files for pretraining using the
permutation language modeling objective.
"""
import json
import os
import random

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

from text_utils import encode_ids
from text_utils import preprocess_text
from text_utils import CLS_ID
from text_utils import SEP_ID
from text_utils import EOD_ID
from text_utils import SEG_ID_P
from text_utils import SEG_ID_Q
from text_utils import SEG_ID_CLS

flags.DEFINE_list(
    'input_file_paths', None, 'Paths to raw text files to be processed.')
flags.DEFINE_string(
    'spiece_model_path', None, 'Path to SentencePiece model file.')
flags.DEFINE_string(
    'output_file_path', 'pretrain.tfrecord', 'Path to the output TFRecord '
    'file.')
flags.DEFINE_bool(
    'use_eod', True, 'Whether to add the special token EOD to the end of '
    'document.')
flags.DEFINE_integer(
    'batch_size', 2, 'Number of sequences in a batch.')
flags.DEFINE_integer(
    'seq_len', 512, 'Length of sequence in a batch.')
flags.DEFINE_bool(
    'bi_data', True, 'Whether to process text bidirectionally.')
flags.DEFINE_integer(
    'reuse_len', 256, 'Number of tokens that can be reused as memory.')
flags.DEFINE_bool(
    'uncased', False, 'Whether to use uncased inputs.')

FLAGS = flags.FLAGS


def _split_a_and_b(token_ids, sent_ids, begin_index, size):
  """Sample seq a from `token_ids`, starting at `begin_index`. seq b either
  follows immediately seq a, or is sampled randomly. Try making sure that seq a
  and seq b starts and ends at sentence boundaries.

  Args:
    token_ids: numpy array of shape [data_len], sequence of token IDs in a
      single batch.
    sent_ids: numpy array of shape [data_len], sequence of sentence IDs in a
      sinlge batch.
    begin_index: int, seq a will be sampled from `token_ids` starting at
      `begin_index`.
    size: int, total length of seq a and seq b.

  Returns:
    seq_a: numpy array of shape [seq_a_len], token IDs of seq a.
    seq_b: numpy array of shape [seq_b_len], token IDs of seq b.
    label: int, integer indicating if seq b immediately follows seq a (1) or is
      sampled randomly (0).
  """
  data_len = token_ids.shape[0]

  end_index = begin_index + 1
  cut_points = []
  while end_index < data_len:
    if sent_ids[end_index] != sent_ids[end_index - 1]:
      if end_index - begin_index >= size:
        break
      cut_points.append(end_index)
    end_index += 1

  a_begin = begin_index
  if len(cut_points) == 0 or random.random() < 0.5:
    # CASE 0: seq b is randomly sampled
    label = 0
    if len(cut_points) == 0:
      a_end = end_index
    else:
      a_end = random.choice(cut_points)

    b_len = max(1, size - (a_end - a_begin))
    b_begin = random.randint(0, data_len - b_len)
    b_end = b_begin + b_len

    # expand seq b in both directions to make sure that `b_begin` and `b_end`
    # sit at sentence boundaries.
    while b_begin > 0 and sent_ids[b_begin - 1] == sent_ids[b_begin]:
      b_begin -= 1
    while b_end < data_len and sent_ids[b_end - 1] == sent_ids[b_end]:
      b_end += 1
  else:
    # CASE 1: seq b follows immediately seq a
    label = 1
    a_end = random.choice(cut_points)
    b_begin = a_end
    b_end = end_index

  # while the total length of seq a and seq b > size,
  # delete tokens one at a time on the right side
  while a_end - a_begin + b_end - b_begin > size:
    if a_end - a_begin > b_end - b_begin:
      a_end -= 1
    else:
      b_end -= 1

  seq_a = token_ids[a_begin:a_end]
  seq_b = token_ids[b_begin:b_end]

  return seq_a, seq_b, label


def _batch_data(token_ids, sent_ids, batch_size):
  """Batch the input sequence of IDs.

  Args:
    token_ids: numpy array of shape [total_num_ids], all token IDs.
    sent_ids: numpy array of shape [total_num_ids], sentence IDs of all tokens
      (alternating 1's and 0's).
    batch_size: int, number of sequences in a batch.

  Returns:
    token_ids: numpy array of shape [batch_size, data_len], batched sequences of
      token IDs.
    sent_ids: numpy array of shape [batch_size, data_len], batched sequences of
      sentence IDs.
  """
  data_len = len(token_ids) // batch_size
  token_ids = token_ids[:batch_size * data_len]
  token_ids = token_ids.reshape(batch_size, data_len)
  sent_ids = sent_ids[:batch_size * data_len]
  sent_ids = sent_ids.reshape(batch_size, data_len)

  return token_ids, sent_ids


def _create_tfrecords(output_file_path,
                      token_ids,
                      sent_ids,
                      batch_size,
                      seq_len,
                      bi_data,
                      reuse_len=256):
  """Write token IDs into TFRecord files.

  Args:
    output_file_path: string, path to the output TFRecord file.
    token_ids: numpy array of shape [total_num_ids], all token IDs.
    sent_ids: numpy array of shape [total_num_ids], sentence IDs of all tokens
      (alternating 1's and 0's).
    batch_size: int, number of sequences in a batch.
    seq_len: int, length of sequence in a batch.
    bi_data: bool, whether to process text bidirectionally.
    reuse_len: int, number of tokens that can be reused as memory.
  """
  if bi_data:
    if batch_size % 2 != 0:
      raise ValueError('`batch_size` must be divisible by 2 for bidirectional '
          'input.')
    fwd_token_ids, fwd_sent_ids = _batch_data(
        token_ids, sent_ids, batch_size // 2)
    bwd_token_ids = fwd_token_ids[:, ::-1]
    bwd_sent_ids = fwd_sent_ids[:, ::-1]
    token_ids = np.vstack([fwd_token_ids, bwd_token_ids])
    sent_ids = np.vstack([fwd_sent_ids, bwd_sent_ids])
  else:
    token_ids, sent_ids = _batch_data(token_ids, sent_ids, batch_size)

  # each sequence in a batch has 2 SEP tokens and 1 CLS token.
  if reuse_len >= seq_len - 3:
    raise ValueError(f'It must hold that `reuse_len < seq_len - 3`, got '
        'reuse_len = {reuse_len}, seq_len = {seq_len}')

  sep_array = np.array([SEP_ID], dtype='int64')
  cls_array = np.array([CLS_ID], dtype='int64')

  i, num_batches = 0, 0
  counts = [0, 0]
  with tf.io.TFRecordWriter(output_file_path) as record_writer:
    while i + seq_len <= token_ids.shape[1]:
      for index in range(batch_size):
        data_a, data_b, label = _split_a_and_b(token_ids[index],
                                               sent_ids[index],
                                               begin_index=i+reuse_len,
                                               size=seq_len-reuse_len-3)

        batch_token_ids = np.concatenate([token_ids[index, i:i+reuse_len],
            data_a, sep_array, data_b, sep_array, cls_array])
        batch_seg_ids = ([SEG_ID_P] * (reuse_len + data_a.shape[0] + 1) +
            [SEG_ID_Q] * (data_b.shape[0] + 1) + [SEG_ID_CLS])

        feature = {'token_ids': tf.train.Feature(
            int64_list=tf.train.Int64List(value=batch_token_ids)),
                   'seg_ids': tf.train.Feature(
            int64_list=tf.train.Int64List(value=batch_seg_ids))}

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        record_writer.write(example.SerializeToString())
        counts[label] += 1
      num_batches += 1

      i += reuse_len
  logging.info(f'Total number of batches: {num_batches}')
  logging.info(f'Sequence pair counts by type: {counts[0]}(0), {counts[1]}(1)')


def _create_data(input_file_paths,
                 spiece_model_path,
                 output_file_path,
                 use_eod,
                 batch_size=4,
                 seq_len=512,
                 bi_data=True,
                 reuse_len=256,
                 uncased=False):
  """Convert input text files into token IDs, shuffle the token-IDs at the file
  level, and finally write the data to TFRecord files.

  Args:
    input_file_paths: list of strings, paths to raw text files to be processed.
    spiece_model_path: string, path to SentencePiece model file.
    output_file_path: string, path to the output TFRecord file.
    use_eod: bool, whether to add the special token EOD to the end of document.
    batch_size: int, number of sequences in a batch.
    seq_len: int, length of sequence in a batch.
    bi_data: bool, whether to process text bidirectionally.
    reuse_len: int, number of tokens that can be reused as memory.
    uncased: bool, whether to use uncased inputs.
  """
  sp = spm.SentencePieceProcessor()
  sp.Load(spiece_model_path)

  # go through the raw text files and convert raw texts into token IDs
  per_file_ids = []
  for input_file_path in input_file_paths:
    token_ids, sent_ids = [], []
    sent_id = True
    with tf.io.gfile.GFile(input_file_path) as f:
      for line in f:
        if not line.strip():
          if use_eod:
            # treat empty line as a sentence with only one token EOD
            sent_id = not sent_id
            curr_sent = [EOD_ID]
          else:
            continue
        else:
          curr_sent = preprocess_text(
              line.strip(), lower=uncased)
          curr_sent = encode_ids(sp, curr_sent)
        token_ids.extend(curr_sent)
        sent_ids.extend([sent_id] * len(curr_sent))
        sent_id = not sent_id

    if len(token_ids) == 0:
      continue

    token_ids = np.array(token_ids, dtype='int64')
    sent_ids = np.array(sent_ids, dtype='bool')
    per_file_ids.append((token_ids, sent_ids))

  # shuffle the token-IDs at the file level
  perm_indices = np.random.permutation(len(per_file_ids))
  token_ids_list, sent_ids_list = [], []
  prev_sent_id = None
  for index in perm_indices:
    token_ids, sent_ids = per_file_ids[index]

    if prev_sent_id is not None and sent_ids[0] == prev_sent_id:
      sent_ids = np.logical_not(sent_ids)

    token_ids_list.append(token_ids)
    sent_ids_list.append(sent_ids)

    prev_sent_id = sent_ids[-1]
  token_ids = np.concatenate(token_ids_list)
  sent_ids = np.concatenate(sent_ids_list)

  # write the data to TFRecord files
  _create_tfrecords(output_file_path,
                    token_ids,
                    sent_ids,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    bi_data=bi_data,
                    reuse_len=reuse_len)


def main(_):
  input_file_paths = FLAGS.input_file_paths
  spiece_model_path = FLAGS.spiece_model_path
  output_file_path = FLAGS.output_file_path
  use_eod = FLAGS.use_eod
  batch_size = FLAGS.batch_size
  seq_len = FLAGS.seq_len
  bi_data = FLAGS.bi_data
  reuse_len = FLAGS.reuse_len
  uncased = FLAGS.uncased

  _create_data(input_file_paths,
               spiece_model_path,
               output_file_path,
               use_eod,
               batch_size=batch_size,
               seq_len=seq_len,
               bi_data=bi_data,
               reuse_len=reuse_len,
               uncased=uncased)


if __name__ == '__main__':
  flags.mark_flag_as_required('input_file_paths')
  flags.mark_flag_as_required('spiece_model_path')
  app.run(main)

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

from text_utils import encode_ids
from text_utils import preprocess_text
from text_utils import CLS_ID
from text_utils import SEP_ID
from text_utils import EOD_ID


flags.DEFINE_list(
    'input_file_paths', None, 'Paths to raw text files to be converted.')
flags.DEFINE_string(
    'spiece_model_path', None, 'Path to SentencePiece model file.')
flags.DEFINE_string(
    'output_file_path', 'pretrain.tfrecord', 'Path to the output tfrecord '
    'file.')    
flags.DEFINE_bool(
    'use_eod', True, 'Whether to add End-Of-Document token (EOD).')
flags.DEFINE_integer(
    'batch_size', 4, 'Batch size.')
flags.DEFINE_integer(
    'seq_len', 512, 'Length of sequence in a batch.')
flags.DEFINE_bool(
    'bi_data', True, 'Whether to process text bidirectionally.')
flags.DEFINE_integer(
    'mask_alpha', 6, 'How many tokens to form a group.')
flags.DEFINE_integer(
    'mask_beta', 1, 'How many tokens to mask within each group.')
flags.DEFINE_integer(
    'reuse_len', 256, 'Number of token that can be reused as memory.')
flags.DEFINE_integer(
    'num_predict', 85, 'Num of tokens to predict.')
flags.DEFINE_bool(
    'uncased', False, 'Use uncased inputs or not.')

FLAGS = flags.FLAGS


def _split_a_and_b(data, sent_ids, begin_idx, tot_len, extend_target=False):
  """Split two segments from `data` starting from the index `begin_idx`."""

  data_len = data.shape[0]
  if begin_idx + tot_len >= data_len:
    return None

  end_idx = begin_idx + 1
  cut_points = []
  while end_idx < data_len:
    if sent_ids[end_idx] != sent_ids[end_idx - 1]:
      if end_idx - begin_idx >= tot_len: break
      cut_points.append(end_idx)
    end_idx += 1

  a_begin = begin_idx
  if len(cut_points) == 0 or random.random() < 0.5:
    label = 0
    if len(cut_points) == 0:
      a_end = end_idx
    else:
      a_end = random.choice(cut_points)

    b_len = max(1, tot_len - (a_end - a_begin))
    # (zihangd): `data_len - 1` to account for extend_target
    b_begin = random.randint(0, data_len - 1 - b_len)
    b_end = b_begin + b_len
    while b_begin > 0 and sent_ids[b_begin - 1] == sent_ids[b_begin]:
      b_begin -= 1
    # (zihangd): `data_len - 1` to account for extend_target
    while b_end < data_len - 1 and sent_ids[b_end - 1] == sent_ids[b_end]:
      b_end += 1

    new_begin = a_end
  else:
    label = 1
    a_end = random.choice(cut_points)
    b_begin = a_end
    b_end = end_idx

    new_begin = b_end

  while a_end - a_begin + b_end - b_begin > tot_len:
    if a_end - a_begin > b_end - b_begin:
      # delete the right side only for the LM objective
      a_end -= 1
    else:
      b_end -= 1

  ret = [data[a_begin: a_end], data[b_begin: b_end], label]

  return ret


def _batchify(data, bsz_per_host, sent_ids=None):
  num_step = len(data) // bsz_per_host
  data = data[:bsz_per_host * num_step]
  data = data.reshape(bsz_per_host, num_step)
  if sent_ids is not None:
    sent_ids = sent_ids[:bsz_per_host * num_step]
    sent_ids = sent_ids.reshape(bsz_per_host, num_step)

  if sent_ids is not None:
    return data, sent_ids
  return data


def _create_tfrecords(output_path,
                     data,
                     bsz_per_host,
                     seq_len,
                     bi_data,
                     sp,
                     mask_alpha=6,
                     mask_beta=1,
                     reuse_len=256,
                     uncased=False,
                     num_predict=85,
                      ):
  data, sent_ids = data[0], data[1]
  num_core_per_host = 1

  num_core = num_core_per_host
  bsz_per_core = bsz_per_host // num_core

  if bi_data:
    assert bsz_per_host % (2 * num_core_per_host) == 0
    fwd_data, fwd_sent_ids = _batchify(data, bsz_per_host // 2, sent_ids)

    fwd_data = fwd_data.reshape(num_core, 1, bsz_per_core // 2, -1)
    fwd_sent_ids = fwd_sent_ids.reshape(num_core, 1, bsz_per_core // 2, -1)

    bwd_data = fwd_data[:, :, :, ::-1]
    bwd_sent_ids = fwd_sent_ids[:, :, :, ::-1]

    data = np.concatenate(
        [fwd_data, bwd_data], 1).reshape(bsz_per_host, -1)
    sent_ids = np.concatenate(
        [fwd_sent_ids, bwd_sent_ids], 1).reshape(bsz_per_host, -1)
  else:
    data, sent_ids = _batchify(data, bsz_per_host, sent_ids)

  record_writer = tf.io.TFRecordWriter(output_path)

  num_batch = 0
  reuse_len = reuse_len

  # [sep] x 2 + [cls]
  assert reuse_len < seq_len - 3

  data_len = data.shape[1]
  sep_array = np.array([SEP_ID], dtype=np.int64)
  cls_array = np.array([CLS_ID], dtype=np.int64)

  i = 0
  while i + seq_len <= data_len:

    all_ok = True
    features = []
    for idx in range(bsz_per_host):
      inp = data[idx, i: i + reuse_len]

      results = _split_a_and_b(
          data[idx],
          sent_ids[idx],
          begin_idx=i + reuse_len,
          tot_len=seq_len - reuse_len - 3,
          extend_target=True)
      if results is None:
        all_ok = False
        break

      # unpack the results
      (a_data, b_data, label) = tuple(results)

      # concatenate data
      cat_data = np.concatenate([inp, a_data, sep_array, b_data,
                                 sep_array, cls_array])
      seg_id = ([0] * (reuse_len + a_data.shape[0]) + [0] +
                [1] * b_data.shape[0] + [1] + [2])
      assert cat_data.shape[0] == seq_len

      feature = {
          "token_ids": tf.train.Feature(
              int64_list=tf.train.Int64List(value=cat_data)),
          "segment_ids": tf.train.Feature(
              int64_list=tf.train.Int64List(value=seg_id)),
      }
      features.append(feature)

    if all_ok:
      assert len(features) == bsz_per_host
      for feature in features:
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        record_writer.write(example.SerializeToString())
      num_batch += 1
    else:
      break

    i += reuse_len

  record_writer.close()

def _create_data(input_paths,
                 sp_path,
                 use_eod,
                 output_path, 
                 bsz_per_host=4,
                 seq_len=512,
                 bi_data=True,
                 mask_alpha=6,
                 mask_beta=1,
                 reuse_len=256,
                 num_predict=85,
                 uncased=False):
  sp = spm.SentencePieceProcessor()
  sp.Load(sp_path)

  input_shards = []
  total_line_cnt = 0
  for input_path in input_paths:
    input_data, sent_ids = [], []
    sent_id, line_cnt = True, 0
    for line in tf.io.gfile.GFile(input_path):
      line_cnt += 1

      if not line.strip():
        if use_eod:
          sent_id = not sent_id
          cur_sent = [EOD_ID]
        else:
          continue
      else:
        cur_sent = preprocess_text(
            line.strip(), lower=uncased)
        cur_sent = encode_ids(sp, cur_sent)

      input_data.extend(cur_sent)
      sent_ids.extend([sent_id] * len(cur_sent))
      sent_id = not sent_id

    if line_cnt == 0:
      continue

    input_data = np.array(input_data, dtype=np.int64)
    sent_ids = np.array(sent_ids, dtype=np.bool)

    total_line_cnt += line_cnt
    input_shards.append((input_data, sent_ids))

  filenames, num_batch = [], 0

  perm_indices = np.random.permutation(len(input_shards))

  input_data_list, sent_ids_list = [], []
  prev_sent_id = None 
  for perm_idx in perm_indices:
    input_data, sent_ids = input_shards[perm_idx]

    if prev_sent_id is not None and sent_ids[0] == prev_sent_id:
      sent_ids = np.logical_not(sent_ids)

    input_data_list.append(input_data)
    sent_ids_list.append(sent_ids)

    prev_sent_id = sent_ids[-1]

  input_data = np.concatenate(input_data_list)
  sent_ids = np.concatenate(sent_ids_list)

  _create_tfrecords(output_path,
                   (input_data, sent_ids),
                   bsz_per_host=bsz_per_host,
                   seq_len=seq_len,
                   bi_data=bi_data,
                   sp=sp,
                   mask_alpha=mask_alpha,
                   mask_beta=mask_beta,
                   reuse_len=reuse_len,
                   uncased=uncased,
                   num_predict=num_predict)

  return input_data, sent_ids
 

def main(_):
  input_file_paths = FLAGS.input_file_paths
  spiece_model_path = FLAGS.spiece_model_path
  use_eod = FLAGS.use_eod
  output_file_path = FLAGS.output_file_path
  batch_size = FLAGS.batch_size
  seq_len = FLAGS.seq_len
  bi_data = FLAGS.bi_data
  mask_alpha = FLAGS.mask_alpha
  mask_beta = FLAGS.mask_beta
  reuse_len = FLAGS.reuse_len
  num_predict = FLAGS.num_predict
  uncased = FLAGS.uncased

  input_data, sent_ids = _create_data(
      input_file_paths,
      spiece_model_path,
      use_eod,
      output_file_path,
      bsz_per_host=batch_size,
      seq_len=seq_len,
      bi_data=bi_data,
      mask_alpha=mask_alpha,
      mask_beta=mask_beta,
      reuse_len=reuse_len,
      num_predict=num_predict,
      uncased=uncased,
    )


if __name__ == '__main__':
  flags.mark_flag_as_required('input_file_paths')
  flags.mark_flag_as_required('spiece_model_path')
  app.run(main)

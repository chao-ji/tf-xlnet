"""Convert IMDB dataset into TFRecord files for training and evaluation."""
import os

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

from text_utils import encode_ids
from text_utils import preprocess_text
from text_utils import SEG_ID_P
from text_utils import SEG_ID_Q
from text_utils import SEG_ID_CLS
from text_utils import SEG_ID_PAD
from text_utils import CLS_ID
from text_utils import SEP_ID


flags.DEFINE_string(
    'spiece_model_path', None, 'Path to SentencePiece model file.')
flags.DEFINE_string(
    'data_path', None, 'Path to directory holding IMDB dataset.')
flags.DEFINE_string(
    'train_output_file_path', './imdb_train.tfrecord', 'Path to the output '
        'file for the train split')
flags.DEFINE_string(
    'dev_output_file_path', './imdb_dev.tfrecord', 'Path to the output file '
        'for the dev split')
flags.DEFINE_integer(
    'seq_len', 512, 'Maximum number of tokens in a sequence.')

FLAGS = flags.FLAGS


def _read_imdb_data(data_dir):
  """Read raw IMDB data.

  Args:
    data_dir: string scalar, path to the directory containing IMDB data.

  Returns:
    instances: list of dicts, IMDB instances.
  """
  instances = []
  for label in ['neg', 'pos']:
    cur_dir = os.path.join(data_dir, label)
    for filename in tf.io.gfile.listdir(cur_dir):
      if not filename.endswith('txt') :
        continue

      path = os.path.join(cur_dir, filename)
      with tf.io.gfile.GFile(path) as f:
        text = f.read().strip().replace("<br />", " ")
      instances.append(
          {'text': text, 'label': label})
  return instances


def _convert_single_instance(instance, label_list, seq_len, sp_model):
  """Converts a single `InputExample` into a single `InputFeatures`.

  Args:
    instance: dict, an IMDB instance.
    label_list: list of strings, sequence-level labels.
    seq_len: int scalar, max number of tokens in a sequence.
    sp_model: an instance of SentencePieceProcessor, sentence piece processor.

  Returns:
    instance: dict with the following entries,
      'token_ids' -> list of integers, token IDs of the text
      'pad_mask' -> list of floats, sequence of 1's and 0's where 1's indicate
        padded (masked) tokens.
      'seg_ids' -> list of integers, sequence of segment IDs.
      'label_ids' -> int scalar, sequence-level label.
  """
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  token_ids = encode_ids(
      sp_model, preprocess_text(instance['text'], lower=False))[:seq_len-2]

  token_ids = token_ids + [SEP_ID, CLS_ID]
  seg_ids = (len(token_ids) - 1) * [SEG_ID_P] + [SEG_ID_CLS]

  pad_mask = [0] * len(token_ids)

  if len(token_ids) < seq_len:
    pad_len = seq_len - len(token_ids)

    token_ids = [0] * pad_len + token_ids
    pad_mask = [1] * pad_len + pad_mask
    seg_ids = [SEG_ID_PAD] * pad_len + seg_ids

  label_id = label_map[instance['label']]
  instance = {'token_ids': token_ids,
             'pad_mask': pad_mask,
             'seg_ids': seg_ids,
             'label_ids': label_id}

  return instance


def _convert_imdb_instances_to_tfrecord(instances,
                                        label_list,
                                        seq_len,
                                        sp_model,
                                        output_filename):
  """Convert IMDB instances to TFRecord file.

  Args:
    instances: list of dict, IMDB instances.
    label_list: list of strings, sequence-level labels.
    seq_len: int scalar, max number of tokens in a sequence.
    sp_model: an instance of SentencePieceProcessor, sentence piece processor.
    output_filename: string scalar, path to the output TFRecord file.
  """
  with tf.io.TFRecordWriter(output_filename) as record_writer:
    for index, instance in enumerate(instances):
      if index % 1000 == 0:
        logging.info(f'Processing instance {index}')

      feature = _convert_single_instance(instance, label_list, seq_len, sp_model)
      feature = {'token_ids': tf.train.Feature(int64_list=
                     tf.train.Int64List(value=list(feature['token_ids']))),
                 'pad_mask': tf.train.Feature(float_list=
                     tf.train.FloatList(value=list(feature['pad_mask']))),
                 'seg_ids': tf.train.Feature(int64_list=
                     tf.train.Int64List(value=list(feature['seg_ids']))),
                 'label_ids': tf.train.Feature(int64_list=
                     tf.train.Int64List(value=[feature['label_ids']]))}

      record_writer.write(tf.train.Example(features=tf.train.Features(
          feature=feature)).SerializeToString())


def main(_):
  spiece_model_path = FLAGS.spiece_model_path
  data_path = FLAGS.data_path
  train_output_file_path = FLAGS.train_output_file_path
  dev_output_file_path = FLAGS.dev_output_file_path
  seq_len = FLAGS.seq_len

  label_list = ['neg', 'pos']
  sp_model = spm.SentencePieceProcessor()
  sp_model.Load(spiece_model_path)

  train_instances = _read_imdb_data(os.path.join(data_path, 'train'))
  np.random.shuffle(train_instances)
  eval_instances = _read_imdb_data(os.path.join(data_path, 'test'))

  _convert_imdb_instances_to_tfrecord(train_instances,
                                      label_list,
                                      seq_len,
                                      sp_model,
                                      train_output_file_path)
  _convert_imdb_instances_to_tfrecord(eval_instances,
                                      label_list,
                                      seq_len,
                                      sp_model,
                                      dev_output_file_path)


if __name__ == '__main__':
  flags.mark_flag_as_required('spiece_model_path')
  flags.mark_flag_as_required('data_path')
  app.run(main)

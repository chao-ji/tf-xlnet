"""Convert IMDB dataset into TFRecord files for training and evaluation."""
import os

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from absl import app
from absl import flags

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
    'max_seq_length', 512, 'Maximum sequence length.')

FLAGS = flags.FLAGS


def _create_examples(data_dir):
  examples = []
  for label in ['neg', 'pos']:
    cur_dir = os.path.join(data_dir, label)
    for filename in tf.io.gfile.listdir(cur_dir):
      if not filename.endswith('txt') :
        continue

      path = os.path.join(cur_dir, filename)
      with tf.io.gfile.GFile(path) as f:
        text = f.read().strip().replace("<br />", " ")
      examples.append(
          {'text_a': text, 'text_b': None, 'label': label})
  return examples


def _tokenize_fn(sp, text):
  text = preprocess_text(text, lower=False)
  return encode_ids(sp, text)


def _convert_single_example(example_index, example, label_list, max_seq_length,
                           _tokenize_fn, sp, use_bert_format):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  if label_list is not None:
    label_map = {}
    for (i, label) in enumerate(label_list):
      label_map[label] = i

  tokens_a = _tokenize_fn(sp, example['text_a'])
  tokens_b = None
  if example['text_b']:
    tokens_b = _tokenize_fn(sp, example['text_b'])

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for two [SEP] & one [CLS] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for one [SEP] & one [CLS] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[:max_seq_length - 2]

  tokens = []
  segment_ids = []
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(SEG_ID_P)
  tokens.append(SEP_ID)
  segment_ids.append(SEG_ID_P)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(SEG_ID_Q)
    tokens.append(SEP_ID)
    segment_ids.append(SEG_ID_Q)

  if use_bert_format:
    tokens.insert(0, CLS_ID)
    segment_ids.insert(0, SEG_ID_CLS)
  else:
    tokens.append(CLS_ID)
    segment_ids.append(SEG_ID_CLS)

  input_ids = tokens

  # The mask has 0 for real tokens and 1 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [0] * len(input_ids)

  # Zero-pad up to the sequence length.
  if len(input_ids) < max_seq_length:
    delta_len = max_seq_length - len(input_ids)
    if use_bert_format:
      input_ids = input_ids + [0] * delta_len
      input_mask = input_mask + [1] * delta_len
      segment_ids = segment_ids + [SEG_ID_PAD] * delta_len
    else:
      input_ids = [0] * delta_len + input_ids
      input_mask = [1] * delta_len + input_mask
      segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  if label_list is not None:
    label_id = label_map[example['label']]
  else:
    label_id = example['label']

  return {'token_ids': input_ids,
          'token_mask': input_mask,
          'segment_ids': segment_ids,
          'label_ids': label_id,
          'is_real_example': True}
  


def _file_based_convert_examples_to_features(examples,
                                             label_list,
                                             max_seq_length,
                                             _tokenize_fn,
                                             sp,
                                             output_file,
                                             num_passes=1):

  writer = tf.io.TFRecordWriter(output_file)
  examples *= num_passes

  for (ex_index, example) in enumerate(examples):

    feature = _convert_single_example(ex_index,
                                      example,
                                      label_list,
                                      max_seq_length,
                                      _tokenize_fn,
                                      sp, 
                                      use_bert_format=False)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = {} 
    features["token_ids"] = create_int_feature(feature['token_ids'])
    features["token_mask"] = create_float_feature(feature['token_mask'])
    features["segment_ids"] = create_int_feature(feature['segment_ids'])
    if label_list is not None:
      features["label_ids"] = create_int_feature([feature['label_ids']])
    else:
      features["label_ids"] = create_float_feature(
          [float(feature['label_ids'])])
    features["is_real_example"] = create_int_feature(
        [int(feature['is_real_example'])])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def main(_):
  spiece_model_path = FLAGS.spiece_model_path
  data_path = FLAGS.data_path
  train_output_file_path = FLAGS.train_output_file_path
  dev_output_file_path = FLAGS.dev_output_file_path
  max_seq_length = FLAGS.max_seq_length

  label_list = ['neg', 'pos']
  sp = spm.SentencePieceProcessor()
  sp.Load(spiece_model_path)

  train_examples = _create_examples(os.path.join(data_path, 'train'))
  np.random.shuffle(train_examples)
  eval_examples = _create_examples(os.path.join(data_path, 'test'))  

  _file_based_convert_examples_to_features(train_examples,
                                           label_list,
                                           max_seq_length,
                                           _tokenize_fn,
                                           sp,
                                           train_output_file_path)
  _file_based_convert_examples_to_features(eval_examples,
                                           label_list,
                                           max_seq_length,
                                           _tokenize_fn,
                                           sp,
                                           dev_output_file_path)


if __name__ == '__main__':
  flags.mark_flag_as_required('spiece_model_path')
  flags.mark_flag_as_required('data_path')
  app.run(main)

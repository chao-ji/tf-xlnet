"""Convert official SQuAD 2.0 dataset (`train-v2.0.json` and `dev-v2.0.json`)
into TFRecord files for training and evaluation.
"""
import gc
import json
import os
import pickle

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from absl import app
from absl import flags

from text_utils import normalize_text
from text_utils import encode_ids
from text_utils import encode_pieces
from text_utils import SEG_ID_P
from text_utils import SEG_ID_Q
from text_utils import SEG_ID_CLS
from text_utils import SEG_ID_PAD
from text_utils import CLS_ID
from text_utils import SEP_ID
from text_utils import SPIECE_UNDERLINE

LEN_SEQ_A = 1024
LEN_SEQ_B = 1024


flags.DEFINE_string(
    'spiece_model_path', None, 'Path to SentencePiece model file.')
flags.DEFINE_string(
    'squad_data_path', None, 'Path to directory holding SQuAD dataset.')
flags.DEFINE_string(
    'train_output_file_path', './squad_train.tfrecord', 'Path to the output '
        'file for the train split')
flags.DEFINE_string(
    'dev_output_file_path', './squad_dev.tfrecord', 'Path to the output file '
        'for the dev split')

FLAGS = flags.FLAGS


def _read_squad_instances(input_filename, training):
  """Read raw squad data files.

  Args:
    input_filename: string scalar, filename of raw SQuAD data.
    training: bool scalar, True for train split and False for dev split. 

  Returns:
    instances: list of named tuples, storing SQuAD instances.
  """
  instances = []
  with tf.io.gfile.GFile(input_filename) as f:
    for article in json.load(f)['data']:
      for paragraph in article["paragraphs"]:
        p_text = paragraph['context']
        for qa in paragraph['qas']:
          q_text = qa['question']
          is_impossible = qa['is_impossible']

          if training and not is_impossible:
            answer = qa['answers'][0]
            a_text = answer['text']
            a_start = answer['answer_start']
            instance = {'q_text': q_text,
                        'p_text': p_text,
                        'a_text': a_text,
                        'a_start': a_start,
                        'is_impossible': is_impossible}
          elif training:
            instance = {'q_text': q_text,
                        'p_text': p_text,
                        'is_impossible': is_impossible}
          else:
            instance = {'q_text': q_text,
                        'p_text': p_text,
                        'qa_id': qa['id']}

          instances.append(instance)
  return instances


def _convert_index(index, pos, M=None, is_start=True):
  """Converts index."""
  if index[pos] is not None:
    return index[pos]
  N = len(index)
  rear = pos
  while rear < N - 1 and index[rear] is None:
    rear += 1
  front = pos
  while front > 0 and index[front] is None:
    front -= 1
  assert index[front] is not None or index[rear] is not None
  if index[front] is None:
    if index[rear] >= 1:
      if is_start:
        return 0
      else:
        return index[rear] - 1
    return index[rear]
  if index[rear] is None:
    if M is not None and index[front] < M - 1:
      if is_start:
        return index[front] + 1
      else:
        return M - 1
    return index[front]
  if is_start:
    if index[rear] > index[front] + 1:
      return index[front] + 1
    else:
      return index[rear]
  else:
    if index[rear] > index[front] + 1:
      return index[rear] - 1
    else:
      return index[front]


def _get_doc_spans(target_len, actual_len, stride=128):
  """Compute the start offset(s) and length(s) for the current paragraph.

  Note that if the paragraph is longer than the limit `target_len`, we will
  end up with multiple **spans** with increasing start offsets that cover the
  whole paragraph. 

  Args:
    target_len: int scalar, num of tokens allocated for the paragraph.
    actual_len: int scalar, the actual num of tokens in the paragraph.
    stride: int scalar, the distances between the start offsets if there are
      multiple spans.

  Returns:
    doc_spans: list of 2-tuples, storing the start offset and lengths for the
      spans.
  """
  doc_spans = []
  start_offset = 0
  while start_offset < actual_len:
    curr_len = actual_len - start_offset
    if curr_len > target_len:
      curr_len = target_len
    doc_spans.append((start_offset, curr_len))
    if start_offset + curr_len == actual_len:
      break
    start_offset += min(curr_len, stride)
  return doc_spans


def _find_lcs(lcs,
              trace,
              len_seq_a,
              len_seq_b,
              p_text,
              token_concat,
              uncased,
              max_dist):
  """
  """
  lcs.fill(0)
  trace.clear()

  for i in range(len_seq_a):
    for j in range(i - max_dist, i + max_dist):
      if j >= len_seq_b or j < 0:
        continue

      if i > 0:
        lcs[i, j] = lcs[i - 1, j]
        trace[(i, j)] = 0
      if j > 0 and lcs[i, j - 1] > lcs[i, j]:
        lcs[i, j] = lcs[i, j - 1]
        trace[(i, j)] = 1

      prev_lcs = lcs[i - 1, j - 1] if i > 0 and j > 0 else 0
      if normalize_text(p_text[i], lower=uncased, remove_space=False
          ) == token_concat[j] and prev_lcs + 1 > lcs[i, j]:
        lcs[i, j] = prev_lcs + 1
        trace[(i, j)] = 2


def _find_answer_token_positions(lcs, p_tokens, instance, uncased, training):
  """
  """
  char_to_token_index = []
  token_start_to_char_index = []
  token_end_to_char_index = []
  char_count = 0

  for i, token in enumerate(p_tokens):
    char_to_token_index.extend([i] * len(token))
    token_start_to_char_index.append(char_count)
    char_count += len(token)
    token_end_to_char_index.append(char_count - 1)

  token_concat = ''.join(p_tokens).replace(SPIECE_UNDERLINE, ' ')
  len_seq_a, len_seq_b= len(instance['p_text']), len(token_concat)

  if len_seq_a > LEN_SEQ_A or len_seq_b > LEN_SEQ_B:
    lcs = np.zeros((max(len_seq_a, LEN_SEQ_A),
                    max(len_seq_b, LEN_SEQ_B)), dtype='float32')
    gc.collect()
  trace = {}

  max_dist = abs(len_seq_a - len_seq_b) + 5
  for _ in range(2):
    _find_lcs(lcs,
              trace,
              len_seq_a,
              len_seq_b,
              instance['p_text'],
              token_concat,
              uncased,
              max_dist)
    if lcs[len_seq_a - 1, len_seq_b - 1] > 0.8 * len_seq_a:
      break
    max_dist *= 2

  text_to_token_concat_char_index = [None] * len_seq_a
  token_concat_to_text_char_index = [None] * len_seq_b

  i, j = len_seq_a - 1, len_seq_b - 1
  while i >= 0 and j >= 0:
    if (i, j) not in trace:
      break
    if trace[(i, j)] == 2:
      text_to_token_concat_char_index[i] = j
      token_concat_to_text_char_index[j] = i
      i -= 1
      j -= 1
    elif trace[(i, j)] == 1:
      j -= 1
    else:
      i -= 1

  tok_start_to_orig_index = []
  tok_end_to_orig_index = []

  for i in range(len(p_tokens)):
    start_chartok_pos = token_start_to_char_index[i]
    end_chartok_pos = token_end_to_char_index[i]
    start_orig_pos = _convert_index(token_concat_to_text_char_index,
                                    start_chartok_pos,
                                    len_seq_a,
                                    is_start=True)
    end_orig_pos = _convert_index(token_concat_to_text_char_index,
                                  end_chartok_pos,
                                  len_seq_a,
                                  is_start=False)
    tok_start_to_orig_index.append(start_orig_pos)
    tok_end_to_orig_index.append(end_orig_pos)

  if not training or instance['is_impossible']:
    token_start_position = -1
    token_end_position = -1
  else:
    start_position = instance['a_start']
    end_position = start_position + len(instance['a_text']) - 1

    start_token_concat_position = _convert_index(
        text_to_token_concat_char_index, start_position, is_start=True)
    token_start_position = char_to_token_index[start_token_concat_position]

    end_token_concat_position = _convert_index(
        text_to_token_concat_char_index, end_position, is_start=False)
    token_end_position = char_to_token_index[end_token_concat_position]

    assert token_start_position <= token_end_position

  return (token_start_position,
          token_end_position,
          tok_start_to_orig_index,
          tok_end_to_orig_index)


def _check_is_max_context(doc_spans, cur_span_index, position):
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span[0] + doc_span[1] - 1
    if position < doc_span[0]:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span[0]
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span[1]
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index

def _convert_squad_instances_to_tfrecord(instances,
                                         sp_model,
                                         record_filename,
                                         uncased=False,
                                         seq_len=512,
                                         q_len=64,
                                         stride=128,
                                         training=False):
  """
  """
  writer = tf.io.TFRecordWriter(record_filename)
  lcs = np.zeros((LEN_SEQ_A, LEN_SEQ_B), dtype='float32')

  fs = []

  for example_index, instance in enumerate(instances):
    if example_index % 1000 == 0:
      print(example_index)
    q_ids = encode_ids(sp_model,
        normalize_text(instance['q_text'], lower=uncased))[:q_len]
    p_tokens = encode_pieces(
        sp_model, normalize_text(instance['p_text'], lower=uncased))
    p_ids = list(map(lambda token: sp_model.PieceToId(token), p_tokens))

    (token_start_position,
     token_end_position,
     tok_start_to_orig_index,
     tok_end_to_orig_index) = _find_answer_token_positions(
        lcs, p_tokens, instance, uncased, training)

    doc_spans = _get_doc_spans(seq_len - len(q_ids) - 3, len(p_ids), stride)

    for doc_span_index, (start_offset, span_len) in enumerate(doc_spans):

      if not training:
        token_is_max_context = {}
        cur_tok_start_to_orig_index = []
        cur_tok_end_to_orig_index = []
        tokens = []
        for i in range(span_len):
          split_token_index = start_offset + i
          cur_tok_start_to_orig_index.append(
              tok_start_to_orig_index[split_token_index])
          cur_tok_end_to_orig_index.append(
              tok_end_to_orig_index[split_token_index])
          is_max_context = _check_is_max_context(
              doc_spans, doc_span_index, split_token_index)
          token_is_max_context[len(tokens)] = is_max_context
          tokens.append(p_ids[split_token_index])
        p_len = len(tokens)

        f = {}
        f['token_is_max_context'] = token_is_max_context
        f['example_index'] = example_index
        f['tok_start_to_orig_index'] = cur_tok_start_to_orig_index
        f['tok_end_to_orig_index'] = cur_tok_end_to_orig_index
        f['paragraph_len'] = p_len

        fs.append(f)

      pad_len = max(seq_len - span_len - len(q_ids) - 3, 0)
      token_ids = np.concatenate([
         p_ids[start_offset:start_offset + span_len],
          [SEP_ID],
          q_ids,
          [SEP_ID, CLS_ID],
          np.zeros(pad_len)]).astype('int64')
      segment_ids = np.concatenate([
          [SEG_ID_P] * (span_len + 1),
          [SEG_ID_Q] * (len(q_ids) + 1),
          [SEG_ID_CLS],
          [SEG_ID_PAD] * pad_len]).astype('int64')

      p_mask = np.concatenate([
          np.zeros(span_len),
          np.ones(len(q_ids) + 2),
          [0],
          np.ones(pad_len)]).astype('float32')

      token_mask = np.concatenate([
          np.zeros(span_len + len(q_ids) + 3),
          np.ones(pad_len)]).astype('float32')

      cls_index = span_len + len(q_ids) + 2

      span_is_impossible = True if hasattr(instance, 'is_impossible'
          ) and instance['is_impossible'] else False

      if not span_is_impossible:
        doc_start = start_offset
        doc_end = start_offset + span_len - 1
        out_of_span = False
        if not (token_start_position >= doc_start and
                token_end_position <= doc_end):
          out_of_span = True
        if out_of_span:
          start_position = 0
          end_position = 0
          span_is_impossible = True
        else:
          doc_offset = 0
          start_position = token_start_position - doc_start + doc_offset
          end_position = token_end_position - doc_start + doc_offset

      if span_is_impossible:
        start_position = cls_index
        end_position = cls_index

      feature = {'token_ids': tf.train.Feature(int64_list=
                     tf.train.Int64List(value=list(token_ids))),
                 'token_mask': tf.train.Feature(float_list=
                     tf.train.FloatList(value=list(token_mask))),
                 'p_mask': tf.train.Feature(float_list=
                     tf.train.FloatList(value=list(p_mask))), 
                 'segment_ids':tf.train.Feature(int64_list=
                     tf.train.Int64List(value=list(segment_ids))),
                 'cls_index': tf.train.Feature(int64_list=
                     tf.train.Int64List(value=[cls_index]))}

      if training:
        feature['start_position'] = tf.train.Feature(int64_list=
            tf.train.Int64List(value=[start_position]))
        feature['end_position'] = tf.train.Feature(int64_list=
            tf.train.Int64List(value=[end_position]))
        feature['is_impossible'] = tf.train.Feature(float_list=
            tf.train.FloatList(value=[span_is_impossible]))

      writer.write(tf.train.Example(features=
          tf.train.Features(feature=feature)).SerializeToString())
  writer.close()

  if not training:
    with open('squad_feature_dev.pickle', 'wb') as f:
      pickle.dump(fs, f)


def main(_):
  spiece_model_path = FLAGS.spiece_model_path
  squad_data_path = FLAGS.squad_data_path 
  train_output_file_path = FLAGS.train_output_file_path
  dev_output_file_path = FLAGS.dev_output_file_path

  sp_model = spm.SentencePieceProcessor()
  sp_model.Load(spiece_model_path)

  train = _read_squad_instances(
      os.path.join(squad_data_path, "train-v2.0.json"), True)
  dev = _read_squad_instances(
      os.path.join(squad_data_path, "dev-v2.0.json"), False)

  with open('dev.pickle', 'wb') as f:
    pickle.dump(dev, f)

  np.random.shuffle(train)
  _convert_squad_instances_to_tfrecord(
      train, sp_model, train_output_file_path, training=True)
  _convert_squad_instances_to_tfrecord(
      dev, sp_model, dev_output_file_path, training=False)


if __name__ == '__main__':
  flags.mark_flag_as_required('spiece_model_path')
  flags.mark_flag_as_required('squad_data_path')
  app.run(main)

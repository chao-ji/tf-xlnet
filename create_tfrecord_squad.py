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
from absl import logging

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
        'file for the train split.')
flags.DEFINE_string(
    'dev_output_file_path', './squad_dev.tfrecord', 'Path to the output file '
        'for the dev split.')
flags.DEFINE_string(
    'feature_file_path', 'squad_feature_dev.pickle', 'Path to the file'
        ' containing sequence features of dev split.')

FLAGS = flags.FLAGS


def _read_squad_instances(input_filename, training):
  """Read raw squad data files.

  Args:
    input_filename: string scalar, filename of raw SQuAD data.
    training: bool scalar, True for train split and False for dev split.

  Returns:
    instances: list of dict, storing SQuAD instances.
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


def _convert_index(map_list, index, len_seq_a=None, is_start=True):
  """Map the index from `p_text` to `token_concat`, or map the index from
  `token_concat` back to `p_text`.

  If `map_list[index]` is not None, then it's exactly the mapped index;
  otherwise, we look for the nearest non-None values on both sides, and return
  the result depending on whether `index` is the *start* or *end*.

  Args:
    map_list: list of integers, where `map_list[index]` is the aligned index of
      the query `index`.
    index: int scalar, the query index.
    len_seq_a: (Optional) int scalar, actual length of `p_text`. Defaults to
      None.
    is_start: (Optional) bool scalar, whether `index` is the "start" or "end" of
      an interval. Defaults to True.

  Returns:
    int scalar, the result of the mapping.
  """
  if map_list[index] is not None:
    return map_list[index]
  len_seq = len(map_list)
  rear = index
  while rear < len_seq - 1 and map_list[rear] is None:
    rear += 1
  front = index
  while front > 0 and map_list[front] is None:
    front -= 1
  assert map_list[front] is not None or map_list[rear] is not None
  if map_list[front] is None:
    if map_list[rear] >= 1:
      if is_start:
        return 0
      else:
        return map_list[rear] - 1
    return map_list[rear]
  if map_list[rear] is None:
    if len_seq_a is not None and map_list[front] < len_seq_a - 1:
      if is_start:
        return map_list[front] + 1
      else:
        return len_seq_a - 1
    return map_list[front]
  if is_start:
    return min(map_list[front] + 1, map_list[rear])
  else:
    return max(map_list[rear] - 1, map_list[front])


def _get_doc_spans(target_len, actual_len, stride=128):
  """Compute the start offset(s) and length(s) for the current paragraph.

  Note that if the paragraph is longer than the limit `target_len`, we will
  end up with multiple **spans** with increasing start offsets that cover the
  whole paragraph.

  Args:
    target_len: int scalar, num of tokens allocated for the paragraph.
    actual_len: int scalar, the actual num of tokens in the paragraph.
    stride: (Optional) int scalar, the distances between the start offsets if
      there are multiple spans. Defaults to 128.

  Returns:
    doc_spans: list of 2-tuples, storing the start offset and lengths for the
      spans.
  """
  doc_spans = []
  start_offset = 0
  while start_offset < actual_len:
    curr_span_len = actual_len - start_offset
    if curr_span_len > target_len:
      curr_span_len = target_len
    doc_spans.append((start_offset, curr_span_len))
    if start_offset + curr_span_len == actual_len:
      break
    # the increment cannot exceed `curr_span_len` to make sure that there is no
    # gap between neighboring spans
    start_offset += min(curr_span_len, stride)
  return doc_spans


def _find_lcs(lcs,
              trace,
              len_seq_a,
              len_seq_b,
              p_text,
              token_concat,
              uncased,
              max_dist):
  """Find the longest common subsequence between `p_text` and `token_concat`.

  Args:
    lcs: numpy array of shape [len_seq_a, len_seq_b], where `lcs[i, j]` is the
      length of the longest common subsequence between `p_text[:i]` and
      `token_concat[:j]`.
    trace: dict, mapping index tuples (i, j) to {0, 1, 2}, where `i` and `j` are
      the char index in `p_text` and `token_concat` respectively, and 0, 1, 2
      indicate UP, LEFT, and DIAGONAL movement when traceback.
    len_seq_a: int scalar, actual length of `p_text`.
    len_seq_b: int scalar, actual length of `token_concat`.
    p_text: string scalar, the raw text of text paragraph.
    token_concat: string scalar, the concatenation of all tokens from `p_text`.
    uncased: bool scalar, whether to use uncased inputs.
    max_dist: int scalar, the maximum distance between the index of the
      corresponding position in `token_concat` for any char in `p_text`, and
      its aligned position in `token_concat`.
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
  """Find the token-based start and end indices of the answer given its
  char-based indices (for training), and find the char-based start and end
  indices of each token (for evaluation).

  Args:
    lcs: numpy array of shape [LEN_SEQ_A, LEN_SEQ_B], where `lcs[i, j]` is the
      length of the longest common subsequence between `p_text[:i]` and
      `token_concat[:j]`.
    p_tokens: list of strings, paragraph tokens.
    instance: dict, a SQuAD instance.
    uncased: bool scalar, whether to use uncased inputs.
    training: bool scalar, True for train split and False for dev split.

  Returns:
    token_start_position: int scalar, token-based start index of the answer.
    token_end_position: int scalar, token-based end index of the answer.
    tok_start_to_orig_index: list of integers (of length `len(p_tokens)`), the
      char-based start index of each token in `p_tokens`.
    tok_end_to_orig_index: list of integers (of length `len(p_tokens)`), the
      char-based end index of each token in `p_tokens`.
  """
  # mapping character index to token index
  char_to_token_index = []
  # mapping token index to start index of token at char level
  token_start_to_char_index = []
  # mapping token index to end index of token at char level
  token_end_to_char_index = []
  char_count = 0

  for i, token in enumerate(p_tokens):
    char_to_token_index.extend([i] * len(token))
    token_start_to_char_index.append(char_count)
    char_count += len(token)
    token_end_to_char_index.append(char_count - 1)

  token_concat = ''.join(p_tokens).replace(SPIECE_UNDERLINE, ' ')
  len_seq_a, len_seq_b = len(instance['p_text']), len(token_concat)

  # increase the size of `lcs` only when a longer sequence is encountered
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


def _check_is_max_context(doc_spans, curr_span_index, position):
  """Check whether `position` in the current span (with index `curr_span_index`)
  has the max context (the lesser of number of tokens on the left and right).

  Args:
    doc_spans: list of 2-tuples, storing the start offset and lengths for the
      spans.
    curr_span_index: int scalar, index of the current span.
    position: int scalar, token index.

  Returns:
    is_max_context: bool scalar, whether `position` has the max context in the
      current span.
  """
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span[0] + doc_span[1] - 1
    if position < doc_span[0] or position > end:
      continue
    num_left_context = position - doc_span[0]
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span[1]
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  is_max_context = curr_span_index == best_span_index
  return is_max_context


def _convert_squad_instances_to_tfrecord(instances,
                                         sp_model,
                                         output_filename,
                                         feature_file_path=None,
                                         uncased=False,
                                         seq_len=512,
                                         q_len=64,
                                         stride=128,
                                         training=False):
  """Convert SQuAD instances to TFRecord file.

  Args:
    instances: list of dict, SQuAD instances.
    sp_model: an instance of SentencePieceProcessor, sentence piece processor.
    output_filename: string scalar, path to the output TFRecord file.
    feature_file_path: (Optional) string scalar, path to the file containing
      sequence features of dev split.
    uncased: (Optional) bool scalar, whether to use uncased inputs. Defaults
      to False.
    seq_len: (Optional) int scalar, number of tokens (paragraph, question, 2 SEP
      and 1 CLS). Defaults to 512.
    q_len: (Optional) int scalar, max number of tokens in the question segment.
      Defaults to 64.
    stride: (Optional) int scalar, the distances between the start offsets if
      there are multiple spans. Defaults to 128.
    training: (Optional) bool scalar, True for train split and False for dev
      split.
  """
  lcs = np.zeros((LEN_SEQ_A, LEN_SEQ_B), dtype='float32')
  feature_list = []

  with tf.io.TFRecordWriter(output_filename) as record_writer:
    for index, instance in enumerate(instances):
      if index % 1000 == 0:
        logging.info(f'Processing instance {index}')
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
        # prepare results needed for evaluation if data is not training split.
        if not training:
          token_is_max_context = {}
          curr_tok_start_to_orig_index = []
          curr_tok_end_to_orig_index = []
          tokens = []
          for i in range(span_len):
            split_token_index = start_offset + i
            curr_tok_start_to_orig_index.append(
                tok_start_to_orig_index[split_token_index])
            curr_tok_end_to_orig_index.append(
                tok_end_to_orig_index[split_token_index])
            token_is_max_context[i] = _check_is_max_context(
                doc_spans, doc_span_index, split_token_index)
            tokens.append(p_ids[split_token_index])

          feature_list.append({'token_is_max_context': token_is_max_context,
                               'instance_index': index,
                               'tok_start_to_orig_index':
                                  curr_tok_start_to_orig_index,
                               'tok_end_to_orig_index':
                                  curr_tok_end_to_orig_index,
                               'paragraph_len': len(tokens)})

        pad_len = max(seq_len - span_len - len(q_ids) - 3, 0)
        token_ids = np.concatenate([
            p_ids[start_offset:start_offset + span_len],
            [SEP_ID],
            q_ids,
            [SEP_ID, CLS_ID],
            np.zeros(pad_len)]).astype('int64')
        seg_ids = np.concatenate([
            [SEG_ID_P] * (span_len + 1),
            [SEG_ID_Q] * (len(q_ids) + 1),
            [SEG_ID_CLS],
            [SEG_ID_PAD] * pad_len]).astype('int64')

        para_mask = np.concatenate([
            np.zeros(span_len),
            np.ones(len(q_ids) + 2),
            [0],
            np.ones(pad_len)]).astype('float32')

        pad_mask = np.concatenate([
            np.zeros(span_len + len(q_ids) + 3),
            np.ones(pad_len)]).astype('float32')

        cls_index = span_len + len(q_ids) + 2

        # span is impossible if question is not answerable in the first place
        span_is_impossible = True if hasattr(instance, 'is_impossible'
            ) and instance['is_impossible'] else False

        # if question is answerable, check if the answer text is entirely within
        # the doc span.
        if not span_is_impossible:
          doc_start = start_offset
          doc_end = start_offset + span_len - 1
          out_of_span = (True if token_start_position < doc_start or
                         token_end_position > doc_end else False)
          if out_of_span:
            start_position = 0
            end_position = 0
            span_is_impossible = True
          else:
            start_position = token_start_position - doc_start
            end_position = token_end_position - doc_start

        if span_is_impossible:
          start_position = cls_index
          end_position = cls_index

        feature = {'token_ids': tf.train.Feature(int64_list=
                       tf.train.Int64List(value=list(token_ids))),
                   'pad_mask': tf.train.Feature(float_list=
                       tf.train.FloatList(value=list(pad_mask))),
                   'para_mask': tf.train.Feature(float_list=
                       tf.train.FloatList(value=list(para_mask))),
                   'seg_ids':tf.train.Feature(int64_list=
                       tf.train.Int64List(value=list(seg_ids))),
                   'cls_index': tf.train.Feature(int64_list=
                       tf.train.Int64List(value=[cls_index]))}

        if training:
          feature['start_position'] = tf.train.Feature(int64_list=
              tf.train.Int64List(value=[start_position]))
          feature['end_position'] = tf.train.Feature(int64_list=
              tf.train.Int64List(value=[end_position]))
          feature['is_impossible'] = tf.train.Feature(float_list=
              tf.train.FloatList(value=[span_is_impossible]))

        record_writer.write(tf.train.Example(features=
            tf.train.Features(feature=feature)).SerializeToString())

    if not training:
      with open(feature_file_path, 'wb') as f:
        pickle.dump(feature_list, f)


def main(_):
  spiece_model_path = FLAGS.spiece_model_path
  squad_data_path = FLAGS.squad_data_path
  train_output_file_path = FLAGS.train_output_file_path
  dev_output_file_path = FLAGS.dev_output_file_path
  feature_file_path = FLAGS.feature_file_path

  sp_model = spm.SentencePieceProcessor()
  sp_model.Load(spiece_model_path)

  train = _read_squad_instances(
      os.path.join(squad_data_path, "train-v2.0.json"), True)
  dev = _read_squad_instances(
      os.path.join(squad_data_path, "dev-v2.0.json"), False)

  np.random.shuffle(train)
  _convert_squad_instances_to_tfrecord(
      train, sp_model, train_output_file_path, training=True)
  _convert_squad_instances_to_tfrecord(
      dev, sp_model, dev_output_file_path, feature_file_path, training=False)


if __name__ == '__main__':
  flags.mark_flag_as_required('spiece_model_path')
  flags.mark_flag_as_required('squad_data_path')
  app.run(main)

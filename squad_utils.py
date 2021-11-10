"""Defines utility functions for evaluating on SQuAD 2.0 dataset."""
import re, pickle
import string
import os
import tensorflow as tf
import collections
import json
import sentencepiece as spm
import unicodedata
import gc
import numpy as np
import math

from text_utils import encode_ids
from text_utils import encode_pieces
from text_utils import SEG_ID_P 
from text_utils import SEG_ID_Q
from text_utils import SEG_ID_CLS
from text_utils import SEG_ID_PAD
from text_utils import CLS_ID
from text_utils import SEP_ID
from text_utils import SPIECE_UNDERLINE


def get_tokens(s):
  if not s:
    return []
  return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
  """Computes f1 score."""
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  # pylint: disable=g-explicit-length-test
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def make_qid_to_has_ans(dataset):
  qid_to_has_ans = {}
  for article in dataset:
    for p in article["paragraphs"]:
      for qa in p["qas"]:
        qid_to_has_ans[qa["id"]] = bool(qa["answers"])
  return qid_to_has_ans

def get_raw_scores(dataset, preds):
  """Gets exact scores and f1 scores."""
  exact_scores = {}
  f1_scores = {}
  for article in dataset:
    for p in article["paragraphs"]:
      for qa in p["qas"]:
        qid = qa["id"]
        gold_answers = [
            a["text"] for a in qa["answers"] if normalize_answer(a["text"])
        ]
        if not gold_answers:
          # For unanswerable questions, only correct answer is empty string
          gold_answers = [""]
        if qid not in preds:
          print("Missing prediction for %s" % qid)
          continue
        a_pred = preds[qid]
        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
  return exact_scores, f1_scores

def _compute_softmax(scores):
  """Computes softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs,
                         qid_to_has_ans):
  """Finds all best threshold."""
  best_exact, exact_thresh, has_ans_exact = find_best_thresh(
      preds, exact_raw, na_probs, qid_to_has_ans)
  best_f1, f1_thresh, has_ans_f1 = find_best_thresh(preds, f1_raw, na_probs,
                                                    qid_to_has_ans)
  main_eval["best_exact"] = best_exact
  main_eval["best_exact_thresh"] = exact_thresh
  main_eval["best_f1"] = best_f1
  main_eval["best_f1_thresh"] = f1_thresh
  main_eval["has_ans_exact"] = has_ans_exact
  main_eval["has_ans_f1"] = has_ans_f1

def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
  """Finds best threshold."""
  num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
  cur_score = num_no_ans
  best_score = cur_score
  best_thresh = 0.0
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  for qid in qid_list:
    if qid not in scores:
      continue
    if qid_to_has_ans[qid]:
      diff = scores[qid]
    else:
      if preds[qid]:
        diff = -1
      else:
        diff = 0
    cur_score += diff
    if cur_score > best_score:
      best_score = cur_score
      best_thresh = na_probs[qid]

  has_ans_score, has_ans_cnt = 0, 0
  for qid in qid_list:
    if not qid_to_has_ans[qid]:
      continue
    has_ans_cnt += 1

    if qid not in scores:
      continue
    has_ans_score += scores[qid]

  return 100.0 * best_score / len(
      scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))




def write_predictions(all_examples,
                      all_features,
                      n_best_size,
                      max_answer_length,
                      output_prediction_file,
                      output_nbest_file,
                      output_null_log_odds_file,
                      orig_data,
                      start_n_top,
                      end_n_top):
  print('examples', len(all_examples))
  print('features', len(all_features))

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature['example_index']].append(feature)


  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for example_index, example in enumerate(all_examples):
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive

    for feature_index, feature in enumerate(features):

      # if we could have irrelevant answers, get the min score of irrelevant
      score_null = min(score_null, feature['cls_logits'])

      for i in range(start_n_top):
        for j in range(end_n_top):
          start_log_prob = feature['start_top_log_probs'][i]
          start_index = feature['start_top_index'][i]

          j_index = i * end_n_top + j

          end_log_prob = feature['end_top_log_probs'][j_index]
          end_index = feature['end_top_index'][j_index]

          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index >= feature['paragraph_len'] - 1:
            continue
          if end_index >= feature['paragraph_len'] - 1:
            continue

          if not feature['token_is_max_context'].get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue

          prelim_predictions.append(
              {'feature_index': feature_index,
               'start_index': start_index,
               'end_index': end_index,
               'start_log_prob': start_log_prob,
               'end_log_prob': end_log_prob})


    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x['start_log_prob'] + x['end_log_prob']),
        reverse=True)

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred['feature_index']]

      tok_start_to_orig_index = feature['tok_start_to_orig_index']
      tok_end_to_orig_index = feature['tok_end_to_orig_index']
      start_orig_pos = tok_start_to_orig_index[pred['start_index']]
      end_orig_pos = tok_end_to_orig_index[pred['end_index']]

      paragraph_text = example['p_text']
      final_text = paragraph_text[start_orig_pos:end_orig_pos + 1].strip()

      if final_text in seen_predictions:
        continue

      seen_predictions[final_text] = True

      nbest.append(
          {'text': final_text,
           'start_log_prob': pred['start_log_prob'],
           'end_log_prob': pred['end_log_prob']})

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          {'text': '', 'start_log_prob': -1e6, 'end_log_prob': -1e6})

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry['start_log_prob'] + entry['end_log_prob'])
      if not best_non_null_entry:
        best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for i, entry in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry['text']
      output["probability"] = probs[i]
      output["start_log_prob"] = entry['start_log_prob']
      output["end_log_prob"] = entry['end_log_prob']
      nbest_json.append(output)

    assert len(nbest_json) >= 1
    assert best_non_null_entry is not None

    score_diff = score_null
    scores_diff_json[example['qa_id']] = score_diff

    all_predictions[example['qa_id']] = best_non_null_entry['text']

    all_nbest_json[example['qa_id']] = nbest_json

  with tf.io.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.io.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

  with tf.io.gfile.GFile(output_null_log_odds_file, "w") as writer:
    writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

  qid_to_has_ans = make_qid_to_has_ans(orig_data)
  exact_raw, f1_raw = get_raw_scores(orig_data, all_predictions)
  out_eval = {}

  find_all_best_thresh(out_eval, all_predictions, exact_raw, f1_raw,
                       scores_diff_json, qid_to_has_ans)

  return out_eval


def run_evaluation(dataset, eval_examples, eval_features,
                   original_data, input_meta_data, model):

  index = 0
  for inputs in dataset:

    input_ids = inputs['token_ids']
    segment_ids = inputs['segment_ids']
    token_mask = inputs['token_mask'][:, None]
    p_mask = inputs['p_mask']
    cls_index = inputs['cls_index']

    (start_top_log_probs,
     start_top_index,
     end_top_log_probs,
     end_top_index,
     cls_logits) = model(
        input_ids, segment_ids, token_mask, p_mask, cls_index, training=False)

    start_top_log_probs = start_top_log_probs.numpy()
    start_top_index = start_top_index.numpy()
    end_top_log_probs = end_top_log_probs.numpy()
    end_top_index = end_top_index.numpy()
    cls_logits = cls_logits.numpy()

    batch_size = start_top_log_probs.shape[0]

    for i in range(batch_size):
      eval_features[index]['start_top_log_probs'] = start_top_log_probs[i].tolist()
      eval_features[index]['start_top_index'] = start_top_index[i].tolist()

      eval_features[index]['end_top_log_probs'] = end_top_log_probs[i].tolist()

      eval_features[index]['end_top_index'] = end_top_index[i].tolist()

      eval_features[index]['cls_logits'] = cls_logits[i].tolist()

      index += 1

  output_prediction_file = os.path.join(input_meta_data["predict_dir"],
                                        "predictions.json")
  output_nbest_file = os.path.join(input_meta_data["predict_dir"],
                                   "nbest_predictions.json")
  output_null_log_odds_file = os.path.join(input_meta_data["predict_dir"],
                                           "null_odds.json")

  results = squad_utils.write_predictions(
      eval_examples, eval_features, input_meta_data["n_best_size"],
      input_meta_data["max_answer_length"], output_prediction_file,
      output_nbest_file, output_null_log_odds_file, original_data,
      input_meta_data["start_n_top"], input_meta_data["end_n_top"])

  return results



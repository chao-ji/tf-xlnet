"""Defines utility functions for evaluating on SQuAD 2.0 dataset."""
import collections
import json
import os

import numpy as np
import tensorflow as tf
from absl import logging

from text_utils import encode_ids
from text_utils import encode_pieces
from text_utils import normalize_answer
from text_utils import SEG_ID_P
from text_utils import SEG_ID_Q
from text_utils import SEG_ID_CLS
from text_utils import SEG_ID_PAD
from text_utils import CLS_ID
from text_utils import SEP_ID
from text_utils import SPIECE_UNDERLINE


def compute_similarity_prediction(gt_ans, pred_ans):
  """Computes the set-similarty based prediction score.

  Both groundtruth and predicted answer will be split up into
  whitespace-separated tokens (two sets of strings). Then we compute the
  precision and recall by checking how similar the predicted set is to the
  groundtruth set (jaccard similarity).

  Args:
    gt_ans: string scalar, groundtruth answer text.
    pred_ans: string scalar, predicted answer text.

  Returns:
    f1: float scalar, f1 score.
  """
  # get space-separated tokens
  get_tokens = lambda s: [] if not s else normalize_answer(s).split()

  # list of word-tokens
  gt_tokens = get_tokens(gt_ans)
  # list of word-tokens
  pred_tokens = get_tokens(pred_ans)

  common = collections.Counter(gt_tokens) & collections.Counter(pred_tokens)
  num_same = sum(common.values())

  if len(gt_tokens) == 0 or len(pred_tokens) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gt_tokens == pred_tokens)
  if num_same == 0:
    return 0
  precision = num_same / len(pred_tokens)
  recall = num_same / len(gt_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def compute_exact_prediction(gt_ans, pred_ans):
  """Compute the whole-text prediction score (1 if predicted text matches
  precisely the groudtruth text and 0 otherwise).

  Args:
    gt_ans: string scalar, groundtruth answer text.
    pred_ans: string scalar, predicted answer text.

  Returns:
    score: int scalar, 1 if correct and 0 if incorrect.
  """
  score = int(normalize_answer(gt_ans) == normalize_answer(pred_ans))
  return score


def get_raw_scores(orig_data, pred_ans_text):
  """Compute predictions scores for each question answer instance using the
  *exact* or *similarity* metric.

  Args:
    orig_data: a list of dicts, dev split of the SQuAD dataset.
    pred_ans_text: dict, mapping qid to predicted text.

  Return:
    exact_scores: dict, mapping qid to whole-text prediction score.
    f1_scores: dict, mapping qid to set-similarty based prediction score.
  """
  exact_scores = {}
  f1_scores = {}
  for article in orig_data:
    for p in article["paragraphs"]:
      for qa in p["qas"]:
        qid = qa["id"]
        gold_answers = [
            a["text"] for a in qa["answers"] if normalize_answer(a["text"])
        ]
        if not gold_answers:
          # For unanswerable questions, only correct answer is empty string
          gold_answers = [""]
        if qid not in pred_ans_text:
          logging.warning("Missing prediction for %s" % qid)
          continue
        a_pred = pred_ans_text[qid]
        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact_prediction(a, a_pred)
                                for a in gold_answers)
        f1_scores[qid] = max(compute_similarity_prediction(a, a_pred)
                             for a in gold_answers)
  return exact_scores, f1_scores


def find_best_score(pred_ans_text,
                    scores,
                    pred_answerability,
                    gt_answerability):
  """Find the best score of the prediction accuracy for two tasks.

  1. The answer text prediction task, i.e. percentage of correct text
    predictions (exact metric or set similarity based metric) out of all
    predictions.
  2. The answerability prediction task, i.e. percentage of correct answerability
    predictions out of all answerable questions.

  Args:
    pred_ans_text: dict, mapping qid to predicted text.
    scores: dict, mapping qid to prediction score.
    pred_answerability: dict, mapping qid to predicted answerability score (
      lower value means higher answerability).
    gt_answerability: dict, mapping qid to groundtruth answerability (1 for
      answerable, 0 for unanswerable).

  Return:
    accuracy_text: float scalar, accuracy of text prediction task.
    best_thresh: float scalar, best answerability threshold for text prediction
      task.
    accuracy_answerability: float scalar, accuracy of answerability task.
  """
  # start with the thresh that predicts all instances unanswerable, so the total
  # number of correct predictions is just the number of unanswerable questions.
  num_unanswerable = sum(1 for k in gt_answerability if not gt_answerability[k])
  best_num_correct = curr_num_correct = num_unanswerable
  best_thresh = None

  # then iterate through the predictions in ascending order of
  # `pred_answerability`, from answerable to unanswerable, i.e. 'lowering' the
  # thresh to allow for more 'answerable' predictions.
  qid_list = sorted(pred_answerability, key=lambda k: pred_answerability[k])

  for qid in qid_list:
    if qid not in scores:
      continue
    if gt_answerability[qid]:
      # add to the number of correct predictions, if it is indeed answerable,
      # and we predict that it is answerable
      delta = scores[qid]
    else:
      # if it is indeed unanswerable, we subtract 1 from the current count of
      # correct predictions, if we mistakenly predict that it is answerable
      if pred_ans_text[qid]:
        delta = -1
      else:
        delta = 0
    curr_num_correct += delta
    if curr_num_correct > best_num_correct:
      best_num_correct = curr_num_correct
      best_thresh = pred_answerability[qid]

  num_correct_answerability, num_total_answerability = 0, 0
  for qid in qid_list:
    if not gt_answerability[qid]:
      continue
    num_total_answerability += 1

    if qid not in scores:
      continue
    num_correct_answerability += scores[qid]

  accuracy_text = best_num_correct / len(scores)
  accuracy_answerability = num_correct_answerability / num_total_answerability

  return accuracy_text, best_thresh, accuracy_answerability


def postprocess_predictions(eval_feature_list,
                            n_best_size,
                            max_ans_len,
                            predict_dir,
                            orig_data,
                            start_n_top,
                            end_n_top):
  """
  Args:
    eval_feature_list: list of dict, each dict contains prediction results and
      other data related to a question answer instance.
    n_best_size: int scalar, number of best scoring predictions.
    max_ans_len: int scalar, max length of answer text.
                            output_prediction_file,
                            output_nbest_file,
                            output_null_log_odds_file,
    orig_data: a list of dicts, dev split of the SQuAD dataset.
    start_n_top: int scalar, the number of top-scoring predictions for start
      position.
    end_n_top: int scalar, the number of top-scoring predictions for end
      position.

  Returns:
    results: dict with the following entries
      'best_exact' -> float scalar, accuracy of exact text prediction
      'best_exact_thresh' -> float scalar, best threshold of exact text
        prediction
      'has_ans_exact' -> float scalar, accuracy of exact answerability
        prediction
      'best_f1' -> float scalar, accuracy of set similarity based text
        prediction
      'best_f1_thresh': float scalar, best threshold of set similarity based
        text prediction
      'has_ans_f1': float scalar, accuracy of set similarity based answerability
        prediction
  """
  eval_instance_list = []
  for i in range(len(orig_data)):
    for j in range(len(orig_data[i]['paragraphs'])):
      for k in range(len(orig_data[i]['paragraphs'][j]['qas'])):
        q_text = orig_data[i]['paragraphs'][j]['qas'][k]['question']
        p_text = orig_data[i]['paragraphs'][j]['context']
        qa_id = orig_data[i]['paragraphs'][j]['qas'][k]['id']
        z = {'q_text': q_text, 'p_text': p_text, 'qa_id': qa_id}
        eval_instance_list.append(z)

  # one instance (paragraph) may correspond to multiple features (seq spans)
  instance_index_to_features = collections.defaultdict(list)
  for feature in eval_feature_list:
    instance_index_to_features[feature['instance_index']].append(feature)

  pred_ans_text = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  pred_answerability = collections.OrderedDict()

  # iterate through question answer instances
  for instance_index, instance in enumerate(eval_instance_list):
    features = instance_index_to_features[instance_index]

    # predictions over {all sequence spans} X {start indices} X {end indices}
    # for the current question answer instance
    prelim_predictions = []

    # lower score means higher answerability
    answerability_score = 1000000

    # iterate through predictions over all sequence spans (features)
    # corresponding to the same instance
    for feature_index, feature in enumerate(features):
      answerability_score = min(answerability_score, feature['cls_logits'])

      # iterate through predictions overl all combinations of start and end
      # indices
      for i in range(start_n_top):
        for j in range(end_n_top):
          start_log_prob = feature['start_top_log_probs'][i]
          start_index = feature['start_top_index'][i]
          end_log_prob = feature['end_top_log_probs'][i*end_n_top+j]
          end_index = feature['end_top_index'][i*end_n_top+j]

          if not (start_index <= end_index < feature['paragraph_len'] - 1
              and end_index - start_index + 1 <= max_ans_len
              and feature['token_is_max_context'].get(start_index, False)):
            continue

          prelim_predictions.append(
              {'feature_index': feature_index,
               'start_index': start_index,
               'end_index': end_index,
               'start_log_prob': start_log_prob,
               'end_log_prob': end_log_prob})

    # sort predictions in descending order of the sum of start and end logprobs
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda pred: (pred['start_log_prob'] + pred['end_log_prob']),
        reverse=True)

    seen_predictions = set()
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred['feature_index']]

      # convert token-based start and end index to char-based for final
      # evaluation
      tok_start_to_orig_index = feature['tok_start_to_orig_index']
      tok_end_to_orig_index = feature['tok_end_to_orig_index']
      start_orig_pos = tok_start_to_orig_index[pred['start_index']]
      end_orig_pos = tok_end_to_orig_index[pred['end_index']]

      paragraph_text = instance['p_text']
      final_text = paragraph_text[start_orig_pos:end_orig_pos + 1].strip()

      if final_text in seen_predictions:
        continue

      seen_predictions.add(final_text)

      nbest.append(
          {'text': final_text,
           'start_log_prob': pred['start_log_prob'],
           'end_log_prob': pred['end_log_prob']})

    # In very rare edge cases we could have no valid predictions. So we
    # just create a dummpy prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          {'text': '', 'start_log_prob': -1e6, 'end_log_prob': -1e6})

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry['start_log_prob'] + entry['end_log_prob'])
      if not best_non_null_entry:
        best_non_null_entry = entry

    # compute probabilities over all valid predictions
    total_scores = np.array(total_scores)
    exp = np.exp(total_scores - total_scores.max())
    probs = (exp / exp.sum()).tolist()

    nbest_json = []
    for i, entry in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry['text']
      output["probability"] = probs[i]
      output["start_log_prob"] = entry['start_log_prob']
      output["end_log_prob"] = entry['end_log_prob']
      nbest_json.append(output)

    pred_answerability[instance['qa_id']] = answerability_score
    pred_ans_text[instance['qa_id']] = best_non_null_entry['text']
    all_nbest_json[instance['qa_id']] = nbest_json

  if not tf.io.gfile.exists(predict_dir):
    tf.io.gfile.mkdir(predict_dir)

  output_prediction_file = os.path.join(predict_dir, 'predictions.json')
  output_nbest_file = os.path.join(predict_dir, 'nbest_predictions.json')
  output_null_log_odds_file = os.path.join(predict_dir, 'null_odds.json')
  with tf.io.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(pred_ans_text, indent=4) + "\n")
  with tf.io.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
  with tf.io.gfile.GFile(output_null_log_odds_file, "w") as writer:
    writer.write(json.dumps(pred_answerability, indent=4) + "\n")

  gt_answerability = {}
  for article in orig_data:
    for p in article["paragraphs"]:
      for qa in p["qas"]:
        gt_answerability[qa["id"]] = bool(qa["answers"])

  exact_raw, setsim_raw = get_raw_scores(orig_data, pred_ans_text)

  exact_acc_text, exact_best_thresh, exact_acc_ans = find_best_score(
      pred_ans_text, exact_raw, pred_answerability, gt_answerability)
  setsim_acc_text, setsim_best_thresh, setsim_acc_ans = find_best_score(
      pred_ans_text, setsim_raw, pred_answerability, gt_answerability)

  results = {'best_exact': exact_acc_text,
             'best_exact_thresh': exact_best_thresh,
             'has_ans_exact': exact_acc_ans,
             'best_f1': setsim_acc_text,
             'best_f1_thresh': setsim_best_thresh,
             'has_ans_f1': setsim_acc_ans}

  return results

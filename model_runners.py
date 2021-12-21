"""Defines classes and functions for pretraining/finetuning and evaluating XLNet
models.
"""
import re
import os

import numpy as np
import tensorflow as tf
from absl import logging

import squad_utils


class XLNetTrainer(object):
  """Base class of XLNet trainers. Defines instance methods for printing logging
  info and saving checkpoints.
  """
  def __init__(self,
               ckpt_dir,
               model_prefix,
               num_iterations,
               persist_per_iterations,
               log_per_iterations=100):
    """Constructor.

    Args:
      ckpt_dir: string scalar, the path to the directory where the checkpoint
        files will be written to or loaded from.
      model_prefix: string scalar, prefix of checkpoint files.
      num_iterations: int scalar, num of iterations that the model will be
        trained for.
      persist_per_iterations: save weights to checkpoint files every
        `persist_per_iterations` iterations.
      log_per_iterations: (Optional) int scalar, print log info every
        `log_per_iterations` iterations. Defaults to 100.
    """
    self._ckpt_dir = ckpt_dir
    self._model_prefix = model_prefix
    self._num_iterations = num_iterations
    self._persist_per_iterations = persist_per_iterations
    self._log_per_iterations = log_per_iterations

  def log(self, step, loss, lr, ckpt):
    """Log training progress and save model checkpoints.

    Args:
      step: int scalar, the global step.
      loss: float scalar, total loss.
      lr: float scalar, current learning rate.
      ckpt: a tf.train.Checkpoint instance, save or load weights to/from
        checkpoint file.

    Returns:
      done: bool scalar, whether training is finished.
    """
    if step % self._log_per_iterations == 0:
      logging.info(f'global_step: {step}, loss, {loss:9.6f}, learning_rate:'
          f' {lr:9.6f}')
    if step % self._persist_per_iterations == 0:
      logging.info(f'Saveing checkpoint at global step {step} ...')
      self.save_model(ckpt)
    done = step == self._num_iterations
    return done

  def save_model(self, ckpt):
    """Save model checkpoint files.

    Args:
      ckpt: a tf.train.Checkpoint instance, save or load weights to/from
        checkpoint file.
    """
    ckpt.save(os.path.join(self._ckpt_dir, self._model_prefix))


class XLNetPretrainer(XLNetTrainer):
  """Pretrains an XLNet model on permutation language modeling task."""
  def __init__(self,
               ckpt_dir,
               num_iterations,
               persist_per_iterations,
               log_per_iterations=100,
               model_prefix='xlnet.ckpt'):
    """Constructor.

    Args:
      ckpt_dir: string scalar, the path to the directory where the checkpoint
        files will be written to or loaded from.
      num_iterations: int scalar, num of iterations that the model will be
        trained for.
      persist_per_iterations: save weights to checkpoint files every
        `persist_per_iterations` iterations.
      log_per_iterations: (Optional) int scalar, print log info every
        `log_per_iterations` iterations. Defaults to 100.
      model_prefix: (Optional) string scalar, prefix of checkpoint files.
        Defaults to 'xlnet.ckpt'.
    """
    super(XLNetPretrainer, self).__init__(ckpt_dir,
                                          model_prefix,
                                          num_iterations,
                                          persist_per_iterations,
                                          log_per_iterations)

  def train(self, model, dataset, optimizer, ckpt):
    """Performs training iterations.

    Args:
      model: an instance of PretrainingXLNet, XLNet purposed for pretraining.
      dataset: a tf.data.Dataset instance, the input data generator.
      optimizer: a tf.keras.optimizer.Optimizer instance, applies gradient
        updates.
      ckpt: a tf.train.Checkpoint instance, save or load weights to/from
        checkpoint file.
    """
    @tf.function
    def train_step(token_ids,
                   seg_ids,
                   perm_mask,
                   target_mapping,
                   target,
                   target_mask,
                   mems):
      """Performs a single training step on a minibatch.

      Args:
        token_ids: int tensor of shape [batch_size, seq_len], sequences of token
          IDs.
        seg_ids: int tensor of shape [batch_size, seq_len], segment ids where
          `seg_ids[b]` is a vector of segment IDs for each token in `token_ids`.
        perm_mask: float tensor of shape [batch_size, seq_len, seq_len],
          permutation mask where the `i`th token cannot attend the `j`th token
          if `perm_mask[b, i, j] = 1`.
        target_mapping: float tensor of shape [batch_size, num_predict, seq_len]
          , where `target_mapping[b, i]` is the one-hot encoding of the index of
          the prediction target for the `i` prediction task (out of
          `num_predict`). May be zero-padded in the 2nd dimension.
        target: int tensor of shape [batch_size, num_predict], the token IDs of
          the prediction targets. May be zero-padded in the 2nd dimension.
        target_mask: float tensor of shape [batch_size, num_predict], vectors
          indicating if an entry in `target` is the actual prediction target (1)
          or padded value (0).
        mems: float tensor of shape [batch_size, stack_size, m_seq_len
          , hidden_size], encodings of the memory sequences from the previous
          block.

      Returns:
        loss: float scalar tensor, the loss.
        step: int scalar tensor, the global step.
        lr: float scalar tensor, the learning rate.
        mems: float tensor of shape [batch_size, stack_size, m_seq_len,
          hidden_size], updated memory sequences.
      """
      with tf.GradientTape() as tape:
        logits, mems = model(
            token_ids, seg_ids, perm_mask, target_mapping, mems)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=logits)
        loss = tf.reduce_sum(loss * target_mask) / tf.reduce_sum(target_mask)

      grads = tape.gradient(loss, model.trainable_weights)
      grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      step = optimizer.iterations
      lr = optimizer.learning_rate(step)
      return loss, step - 1, lr, mems

    batch_size = dataset.element_spec['token_ids'].shape[0]
    mems = tf.zeros([batch_size,
                     model._stack_size,
                     model._mem_len,
                     model._hidden_size], dtype='float32')

    latest_ckpt = tf.train.latest_checkpoint(self._ckpt_dir)
    if latest_ckpt:
      logging.info('Restoring from checkpoint: %s ...' % latest_ckpt)
      ckpt.restore(latest_ckpt)
    else:
      logging.info('Training from scratch...')

    for item in dataset:
      token_ids = item['token_ids']
      seg_ids = item['seg_ids']
      perm_mask = item['perm_mask']
      target_mapping = item['target_mapping']
      target = item['target']
      target_mask = item['target_mask']

      loss, step, lr, mems = train_step(token_ids,
                                        seg_ids,
                                        perm_mask,
                                        target_mapping,
                                        target,
                                        target_mask,
                                        mems)
      if self.log(step.numpy(), loss.numpy(), lr.numpy(), ckpt):
        break


class XLNetQuestionAnswerTrainer(XLNetTrainer):
  """Fine-tunes XLNet for question answering task."""
  def __init__(self,
               lr_layer_decay_rate,
               pretrain_model_path,
               ckpt_dir,
               num_iterations,
               persist_per_iterations,
               log_per_iterations=100,
               model_prefix='squad.ckpt'):
    """Constructor.

    Args:
      lr_layer_decay_rate: float scalar, the decay rate of learning rate for
        applying gradient descent for weights in different layers.
      pretrain_model_path: srting scalar, path to the XLNet model pretrained for
        permutation language modeling task.
      ckpt_dir: string scalar, the path to the directory where the checkpoint
        files of the fine-tuned XLNet model will be written to.
      num_iterations: int scalar, num of iterations that the model will be
        trained for.
      persist_per_iterations: save weights to checkpoint files every
        `persist_per_iterations` iterations.
      log_per_iterations: (Optional) int scalar, print log info every
        `log_per_iterations` iterations. Defaults to 100.
      model_prefix: (Optional) string scalar, prefix of checkpoint files.
        Defaults to 'squad.ckpt'.
    """
    super(XLNetQuestionAnswerTrainer, self).__init__(
        ckpt_dir,
        model_prefix,
        num_iterations,
        persist_per_iterations,
        log_per_iterations)
    self._lr_layer_decay_rate = lr_layer_decay_rate
    self._pretrain_model_path = pretrain_model_path

  def train(self, model, dataset, optimizer, ckpt):
    """Performs training iterations.

    Args:
      model: an instance of QuestionAnswerXLNet, XLNet model for question
        answering task.
      dataset: a tf.data.Dataset instance, the input data generator.
      optimizer: a tf.keras.optimizer.Optimizer instance, applies gradient
        updates.
      ckpt: a tf.train.Checkpoint instance, load weights from a pretrained XLNet
        model on permutation language modeling task and save weights of
        fine-tuned XLNet model.
    """
    @tf.function
    def train_step(token_ids,
                   seg_ids,
                   pad_mask,
                   para_mask,
                   cls_index,
                   start_position,
                   end_position,
                   is_impossible):
      """Performs a single training step on a minibatch.

      Args:
        token_ids: int tensor of shape [batch_size, seq_len], sequences of token
          IDs.
        seg_ids: int tensor of shape [batch_size, seq_len], sequences of segment
          IDs (paragraph, question, CLS, and padded tokens).
        pad_mask: float tensor of shape [batch_size, seq_len], sequences of 1's
          and 0's where 1's indicate padded (masked) tokens.
        para_mask: float tensor of shape [batch_size, seq_len], sequences of 1's
          and 0's where 1's indicate non-paragraph (masked) tokens.
        cls_index: int tensor of shape [batch_size], indices of the CLS tokens.
        start_position: int tensor of shape [batch_size], token-based start
          indices of answer text.
        end_position: int tensor of shape [batch_size], token-based end indices
          of answer text.
        is_impossible: bool tensor of shape [batch_size], the binary
          classification labels.

      Returns:
        loss: float scalar tensor, the loss.
        step: int scalar tensor, the global step.
        lr: float scalar tensor, the learning rate.
      """
      with tf.GradientTape() as tape:
        start_logits, end_logits, cls_logits = model(token_ids,
                                                     seg_ids,
                                                     pad_mask,
                                                     para_mask,
                                                     cls_index,
                                                     start_position,
                                                     training=True)
        seq_len = tf.shape(start_logits)[1]
        start_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=start_position, logits=start_logits))
        end_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=end_position, logits=end_logits))
        cls_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=is_impossible, logits=cls_logits))
        loss = (start_loss + end_loss + cls_loss) * 0.5

      grads = tape.gradient(loss, model.trainable_weights)
      grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)

      n_layer = 0
      lr_layer_decay_rate = self._lr_layer_decay_rate
      for i in range(len(grads)):
        m = re.search(r'decoder_layer_(\d+?)', model.trainable_weights[i].name)
        if m is not None:
          n_layer = max(n_layer, int(m.group(1)) + 1)
      for i in range(len(grads)):
        for l in range(n_layer):
          if 'decoder_layer_{}'.format(l) in model.trainable_weights[i].name:
            abs_rate = lr_layer_decay_rate ** (n_layer - 1 - l)
            grads[i] *= abs_rate
            break

      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      step = optimizer.iterations
      lr = optimizer.learning_rate(step)
      return loss, step - 1, lr

    ckpt.restore(self._pretrain_model_path).expect_partial()
    logging.info('Loaded checkpoint %s' % self._pretrain_model_path)

    for item in dataset:
      token_ids = item['token_ids']
      seg_ids = item['seg_ids']
      para_mask = item['para_mask']
      pad_mask = item['pad_mask'][:, tf.newaxis]
      cls_index = item['cls_index']
      start_position = item['start_position']
      end_position = item['end_position']
      is_impossible = item['is_impossible']

      loss, step, lr = train_step(token_ids,
                                  seg_ids,
                                  pad_mask,
                                  para_mask,
                                  cls_index,
                                  start_position,
                                  end_position,
                                  is_impossible)
      if self.log(step.numpy(), loss.numpy(), lr.numpy(), ckpt):
        break


class XLNetClassificationTrainer(XLNetTrainer):
  """Fine-tunes XLNet for sequence classification task."""
  def __init__(self,
               pretrain_model_path,
               ckpt_dir,
               num_iterations,
               persist_per_iterations,
               log_per_iterations=100,
               model_prefix='cls.ckpt'):
    """Constructor.

    Args:
      pretrain_model_path: srting scalar, path to the XLNet model pretrained for
        permutation language modeling task.
      ckpt_dir: string scalar, the path to the directory where the checkpoint
        files of the fine-tuned XLNet model will be written to.
      num_iterations: int scalar, num of iterations that the model will be
        trained for.
      persist_per_iterations: save weights to checkpoint files every
        `persist_per_iterations` iterations.
      log_per_iterations: (Optional) int scalar, print log info every
        `log_per_iterations` iterations. Defaults to 100.
      model_prefix: (Optional) string scalar, prefix of checkpoint files.
        Defaults to 'cls.ckpt'.
    """
    super(XLNetClassificationTrainer, self).__init__(
        ckpt_dir,
        model_prefix,
        num_iterations,
        persist_per_iterations,
        log_per_iterations)
    self._pretrain_model_path = pretrain_model_path

  def train(self, model, dataset, optimizer, ckpt):
    """Performs training iterations.

    Args:
      model: an instance of ClassificationXLNet.
      dataset: a tf.data.Dataset instance, the input data generator.
      optimizer: a tf.keras.optimizer.Optimizer instance, applies gradient
        updates.
      ckpt: a tf.train.Checkpoint instance, load weights from a pretrained XLNet
        model on permutation language modeling task and save weights of
        fine-tuned XLNet model.
    """
    @tf.function
    def train_step(token_ids, seg_ids, pad_mask, label_ids):
      """Performs a single training step on a minibatch.

      Args:
        token_ids: int tensor of shape [batch_size, seq_len], sequences of token
          IDs.
        seg_ids: int tensor of shape [batch_size, seq_len], sequences of segment
          IDs.
        pad_mask: float tensor of shape [batch_size, seq_len], sequences of 1's
          and 0's where 1's indicate padded (masked) tokens.
        label_ids: int tensor of shape [batch_size], sequence-level labels.

      Returns:
        loss: float scalar tensor, the loss.
        step: int scalar tensor, the global step.
        lr: float scalar tensor, the learning rate.
      """
      with tf.GradientTape() as tape:
        logits = model(token_ids, seg_ids, pad_mask)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_ids, logits=logits)

      grads = tape.gradient(loss, model.trainable_weights)
      grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      step = optimizer.iterations
      lr = optimizer.learning_rate(step)
      return loss, step - 1, lr

    ckpt.restore(self._pretrain_model_path).expect_partial()
    logging.info('Loaded checkpoint %s' % self._pretrain_model_path)

    for item in dataset:
      token_ids = item['token_ids']
      seg_ids = item['seg_ids']
      pad_mask = item['pad_mask'][:, tf.newaxis]
      labels = item['label_ids']

      loss, step, lr = train_step(token_ids, seg_ids, pad_mask, labels)

      if self.log(step.numpy(), loss.numpy().mean(), lr.numpy(), ckpt):
        break


def evaluate_squad_xlnet(model,
                         dataset,
                         eval_feature_list,
                         orig_data,
                         start_n_top,
                         end_n_top,
                         n_best_size,
                         max_ans_len,
                         predict_dir):
  """Evaluate XLNet model for question answering task on SQuAD dataset.

  Args:
    model: an instance of QuestionAnswerXLNet, XLNet model for question
      answering task.
    dataset: a tf.data.Dataset instance, the input data generator.
    eval_feature_list: a list of dicts, where the length dict is the number of
      sequence spans of the paragraph text, and each dict contains the data
      needed to convert the token-based indices to/from char-based indices of
      the answer text.
    orig_data: a list of dicts, dev split of the SQuAD dataset.
    start_n_top: int scalar, the number of top-scoring predictions for start
      position.
    end_n_top: int scalar, the number of top-scoring predictions for end
      position.
    n_best_size: int scalar, number of best scoring predictions.
    max_ans_len: int scalar, max length of answer text.
    predict_dir: string scalar, path to the directory where predictions will be
      written to.

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
  index = 0
  for inputs in dataset:
    token_ids = inputs['token_ids']
    seg_ids = inputs['seg_ids']
    pad_mask = inputs['pad_mask'][:, tf.newaxis]
    para_mask = inputs['para_mask']
    cls_index = inputs['cls_index']

    (start_top_log_probs,
     start_top_index,
     end_top_log_probs,
     end_top_index,
     cls_logits) = model(token_ids,
                         seg_ids,
                         pad_mask,
                         para_mask,
                         cls_index,
                         training=False)

    start_top_log_probs = start_top_log_probs.numpy()
    start_top_index = start_top_index.numpy()
    end_top_log_probs = end_top_log_probs.numpy()
    end_top_index = end_top_index.numpy()
    cls_logits = cls_logits.numpy()

    batch_size = start_top_log_probs.shape[0]
    for i in range(batch_size):
      # [start_n_top]
      eval_feature_list[index]['start_top_log_probs'
          ] = start_top_log_probs[i].tolist()
      # [start_n_top]
      eval_feature_list[index]['start_top_index'] = start_top_index[i].tolist()
      # [start_n_top * end_n_top]
      eval_feature_list[index]['end_top_log_probs'
          ] = end_top_log_probs[i].tolist()
      # [start_n_top * end_n_top]
      eval_feature_list[index]['end_top_index'] = end_top_index[i].tolist()
      # []
      eval_feature_list[index]['cls_logits'] = cls_logits[i].tolist()
      index += 1

  results = squad_utils.postprocess_predictions(eval_feature_list,
                                                n_best_size,
                                                max_ans_len,
                                                predict_dir,
                                                orig_data,
                                                start_n_top,
                                                end_n_top)
  return results


def evaluate_classification_xlnet(model, dataset):
  """Evaluate the accuracy of a classification XLNet model.

  Args:
    model: an instance of ClassificationXLNet, classification XLNet model.
    dataset: a tf.data.Dataset instance, the input data generator.

  Returns:
    correct: int scalar, total number of correct predictions out of `total`
      sequences.
    total: int scalar, total number of sequences.
  """
  correct, total = 0, 0

  for item in dataset:
    token_ids = item['token_ids']
    seg_ids = item['seg_ids']
    pad_mask = item['pad_mask'][:, tf.newaxis]

    logits = model(token_ids, seg_ids, pad_mask)

    logits = logits.numpy()
    labels = item['label_ids'].numpy()

    correct += np.sum(np.equal(np.argmax(logits, axis=-1), labels))
    total += logits.shape[0]

  return correct, total

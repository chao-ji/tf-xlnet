"""
"""
import re
import os

import numpy as np
import tensorflow as tf

import squad_utils

PRETRAIN_MODEL_PREFIX = 'xlnet.ckpt' 
SQUAD_MODEL_PREFIX = 'squad.ckpt' 
CLASSIFICATION_MODEL_PREFIX = 'cls.ckpt'


class XLNetPretrainer(object):
  def __init__(self, model):
    self._model = model

  def train(self,
            dataset,
            optimizer,
            ckpt,
            ckpt_dir,
            num_train_steps,
            persist_per_iterations,
            log_per_iterations=100):
    
    @tf.function
    def train_step(input_ids,
                   seg_ids,
                   perm_mask,
                   target_mapping,
                   target,
                   tgt_mask,
                   mems):
      with tf.GradientTape() as tape:
        logits, mems = self._model(
            input_ids, seg_ids, perm_mask, target_mapping, mems)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=logits)

        lm_loss = tf.reduce_sum(loss * tgt_mask) / tf.reduce_sum(tgt_mask)

      tvars = self._model.trainable_variables
      grads = tape.gradient(lm_loss, tvars)
      clipped, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)

      optimizer.apply_gradients(zip(clipped, tvars))

      step = optimizer.iterations
      lr = optimizer.learning_rate(step)
      return lm_loss, step - 1, lr, mems
 

    batch_size = dataset.element_spec['input_ids'].shape[0]

    mems = tf.zeros([batch_size,
                     self._model._stack_size,
                     self._model._mem_len,
                     self._model._hidden_size], dtype='float32')

    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if latest_ckpt:
      print('Restoring from checkpoint: %s ...' % latest_ckpt)
      ckpt.restore(latest_ckpt)
    else:
      print('Training from scratch...')

    for item in dataset:
      input_ids = item['input_ids']
      seg_ids = item['segment_ids']
      perm_mask = item['perm_mask']

      target_mapping = item['target_mapping']
      target = item['target']
      tgt_mask = item['target_mask']

      loss, step, lr, mems = train_step(input_ids,
                                        seg_ids,
                                        perm_mask,
                                        target_mapping,
                                        target,
                                        tgt_mask,
                                        mems)

      if step.numpy() % log_per_iterations == 0:
        print('global_step: %d, loss, %f, learning_rate: %f' % (
            step.numpy(), loss.numpy(), lr.numpy()))
      if step.numpy() % persist_per_iterations == 0:
        print('Saveing checkpoint at global step %d ...' % step.numpy())
        ckpt.save(os.path.join(ckpt_dir, PRETRAIN_MODEL_PREFIX))

      if step.numpy() == num_train_steps:
        break

    

class XLNetQuestionAnswerTrainer(object):
  def __init__(self, model, lr_layer_decay_rate):
    self._model = model
    self._lr_layer_decay_rate = lr_layer_decay_rate

  def train(self,
            dataset,
            optimizer,
            ckpt,
            ckpt_dir,
            num_train_steps,
            persist_per_iterations,
            log_per_iterations=100):
    """
    """
    def compute_loss(log_probs, positions, seq_len):
      one_hot_positions = tf.one_hot(positions, depth=seq_len, dtype=tf.float32)
      loss = -tf.reduce_sum(one_hot_positions * log_probs, axis=-1)
      loss = tf.reduce_mean(loss)
      return loss

    @tf.function
    def train_step(token_ids,
                   segment_ids,
                   token_mask,
                   p_mask,
                   cls_index,
                   start_positions,
                   end_positions,
                   is_impossible):

      with tf.GradientTape() as tape:
        start_logits, end_logits, cls_logits = self._model(token_ids,
                                                   segment_ids,
                                                   token_mask,
                                                   p_mask,
                                                   cls_index,
                                                   start_positions,
                                                   is_impossible,
                                                   True)

        seq_len = tf.shape(start_logits)[1]
        start_log_probs = tf.nn.log_softmax(start_logits, -1)
        end_log_probs = tf.nn.log_softmax(end_logits, -1)

        start_loss = compute_loss(start_log_probs, start_positions, seq_len)
        end_loss = compute_loss(end_log_probs, end_positions, seq_len)

        total_loss = (start_loss + end_loss) * 0.5
        is_impossible = tf.reshape(is_impossible, [-1])
        regression_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=is_impossible, logits=cls_logits)
        regression_loss = tf.reduce_mean(regression_loss)
        total_loss += regression_loss * 0.5
     
      tvars = self._model.trainable_weights
      grads = tape.gradient(total_loss, tvars)

      clipped, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)

      n_layer = 0

      lr_layer_decay_rate = self._lr_layer_decay_rate #0.75

      for i in range(len(clipped)):
        m = re.search(r"decoder_layer_(\d+?)", tvars[i].name)
        if m is not None:
          n_layer = max(n_layer, int(m.group(1)) + 1)

      for i in range(len(clipped)):
        for l in range(n_layer):
          if "decoder_layer_{}".format(l) in tvars[i].name:
            abs_rate = lr_layer_decay_rate ** (n_layer - 1 - l)
            clipped[i] *= abs_rate
            break

      optimizer.apply_gradients(zip(clipped, tvars))

      step = optimizer.iterations
      lr = optimizer.learning_rate(step)
      return total_loss, step - 1, lr

    for item in dataset:
      token_ids = item['token_ids']
      segment_ids = item['segment_ids']
      p_mask = item['p_mask']
      token_mask = item['token_mask']
      cls_index = item['cls_index']
      start_positions = item['start_position']
      end_positions = item['end_position']
      is_impossible = item['is_impossible']
      token_mask = token_mask[:, tf.newaxis]
    
      loss, step, lr = train_step(token_ids,
                                  segment_ids,
                                  token_mask,
                                  p_mask,
                                  cls_index,
                                  start_positions,
                                  end_positions,
                                  is_impossible)
      if step.numpy() % log_per_iterations == 0:
        print('global_step: %d, loss, %f, learning_rate: %f' % (
            step.numpy(), loss.numpy(), lr.numpy()))

      if step.numpy() % persist_per_iterations == 0:
        print('Saveing checkpoint at global step %d ...' % step.numpy())
        ckpt.save(os.path.join(ckpt_dir, SQUAD_MODEL_PREFIX))

      if step.numpy() == num_train_steps:
        break

    ckpt.save(os.path.join(ckpt_dir, SQUAD_MODEL_PREFIX))


class XLNetClassificationTrainer(object):
  def __init__(self, model):
    self._model = model

  def train(self,
            dataset,
            optimizer,
            ckpt,
            ckpt_dir,
            num_train_steps,
            persist_per_iterations,
            log_per_iterations=100):
    @tf.function
    def train_step(input_ids, segment_ids, input_mask, label_ids):
      with tf.GradientTape() as tape:
        logits = self._model(input_ids, segment_ids, input_mask)

        one_hot_target = tf.one_hot(label_ids, 2, dtype='float32')
        per_example_loss = -tf.reduce_sum(
            tf.nn.log_softmax(logits) * one_hot_target, -1)

      tvars = self._model.trainable_weights

      grads = tape.gradient(per_example_loss, tvars)

      clipped, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)

      optimizer.apply_gradients(zip(clipped, tvars))

      step = optimizer.iterations
      lr = optimizer.learning_rate(step)
      return per_example_loss, step - 1, lr    

    for item in dataset:
      input_ids = item['token_ids']
      segment_ids = item['segment_ids']
      input_mask = item['token_mask'][:, tf.newaxis]

      labels = item['label_ids']

      loss, step, lr = train_step(input_ids, segment_ids, input_mask, labels)

      if step.numpy() % log_per_iterations == 0:
        print('global_step: %d, loss, %f, learning_rate: %f' % (
            step.numpy(), loss.numpy().mean(), lr.numpy()))

      if step.numpy() % persist_per_iterations == 0:
        print('Saveing checkpoint at global step %d ...' % step.numpy())
        ckpt.save(os.path.join(ckpt_dir, CLASSIFICATION_MODEL_PREFIX))

      if step.numpy() == num_train_steps:
        break


    ckpt.save(os.path.join(ckpt_dir, CLASSIFICATION_MODEL_PREFIX))
 

class XLNetQuestionAnswerEvaluator(object):
  def __init__(self, model):
    self._model = model

  def evaluate(self,
               dataset,
               eval_examples,
               eval_features,
               original_data,
               start_n_top,
               end_n_top,
               n_best_size,
               max_answer_length,
               predict_dir):
    """
    """
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
       cls_logits) = self._model(input_ids,
                                 segment_ids,
                                 token_mask,
                                 p_mask,
                                 cls_index,
                                 training=False)
 
      start_top_log_probs = start_top_log_probs.numpy()
      start_top_index = start_top_index.numpy()
      end_top_log_probs = end_top_log_probs.numpy()
      end_top_index = end_top_index.numpy()
      cls_logits = cls_logits.numpy()

      batch_size = start_top_log_probs.shape[0]

      for i in range(batch_size):
        eval_features[index]['start_top_log_probs'
            ] = start_top_log_probs[i].tolist()
        eval_features[index]['start_top_index'] = start_top_index[i].tolist()
        eval_features[index]['end_top_log_probs'
            ] = end_top_log_probs[i].tolist()
        eval_features[index]['end_top_index'] = end_top_index[i].tolist()
        eval_features[index]['cls_logits'] = cls_logits[i].tolist()
        index += 1

    output_prediction_file = os.path.join(predict_dir,
                                          "predictions.json")
    output_nbest_file = os.path.join(predict_dir,
                                     "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(predict_dir,
                                             "null_odds.json")

    results = squad_utils.write_predictions(
        eval_examples, eval_features, n_best_size,
        max_answer_length, output_prediction_file,
        output_nbest_file, output_null_log_odds_file, original_data,
        start_n_top, end_n_top)
    return results


class XLNetClassificationEvaluator(object):
  def __init__(self, model):
    self._model = model

  def evaluate(self, dataset):
    correct = 0
    total = 0
    for item in dataset:
      input_ids = item['token_ids']
      segment_ids = item['segment_ids']
      input_mask = item['token_mask'][:, tf.newaxis]

      logits = self._model(input_ids, segment_ids, input_mask)

      logits = logits.numpy()
      labels = item['label_ids'].numpy()
      masks = item['is_real_example'].numpy()

      real_index = np.where(np.equal(masks, 1))

      correct += np.sum(
        np.equal(
          np.argmax(logits[real_index], axis=-1),
          labels[real_index]))

      total += np.shape(real_index)[-1]

    return correct, total

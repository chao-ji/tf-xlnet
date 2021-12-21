"""Defines learning rate schedule and optimizer."""
import tensorflow as tf


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applys a warmup schedule on a given learning rate decay schedule."""
  def __init__(self,
               init_lr,
               decay_schedule_fn,
               warmup_steps,
               power=1.0):
    """Constructor.

    Args:
      init_lr: float scalar, initial learning rate.
      decay_schedule_fn: callable, learning rate decay function.
      warmup_steps: int scalar, number of warm-up training iterations.
      power: float scalar, parameter used in the learning rate schedule. 
    """
    super(WarmupSchedule, self).__init__()
    self._init_lr = init_lr
    self._warmup_steps = warmup_steps
    self._power = power
    self._decay_schedule_fn = decay_schedule_fn

  def __call__(self, step):
    """Computes learning rate for current step.

    Args:
      step: int scalar tensor, the global step.

    Returns:
      float scalar tensor, learning rate for the current step.
    """ 
    global_step_float = tf.cast(step, tf.float32)
    warmup_steps_float = tf.cast(self._warmup_steps, tf.float32)
    warmup_percent_done = global_step_float / warmup_steps_float
    warmup_learning_rate = (
        self._init_lr *
        tf.math.pow(warmup_percent_done, self._power))
    lr = tf.cond(
        global_step_float < warmup_steps_float,
        lambda: warmup_learning_rate,
        lambda: self._decay_schedule_fn(step - self._warmup_steps))
    return lr


def create_optimizer(init_lr,
                     num_train_steps,
                     num_warmup_steps,
                     min_lr_ratio,
                     adam_epsilon=1e-8):
  """Create Adam optimizer.

  Args:
    init_lr: float scalar, initial learning rate.
    num_train_steps: int scalar, number of training iterations.
    num_warmup_steps: int scalar, number of warm-up training iterations.
    min_lr_ratio: float scalar, the final learning rate will be
      `min_lr_ratio * init_lr`.
    adam_epsilon: float scalar, the small values used in Adam optimizer.

  Returns:
    optimizer: an instance of tf.keras.optimizers.Adam.
  """
  learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=init_lr,
      decay_steps=num_train_steps - num_warmup_steps,
      end_learning_rate=init_lr * min_lr_ratio)
  if num_warmup_steps:
    learning_rate_fn = WarmupSchedule(
        initial_learning_rate=init_lr,
        decay_schedule_fn=learning_rate_fn,
        warmup_steps=num_warmup_steps)
  optimizer = tf.keras.optimizers.Adam(
      learning_rate=learning_rate_fn, epsilon=adam_epsilon)
  return optimizer

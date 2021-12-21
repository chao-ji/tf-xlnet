"""Builds & runs an XLNet model purposed for sequence classification task on
IMDB dataset.
"""
import sentencepiece as spm
import tensorflow as tf
from absl import app
from absl import flags

from commons.dataset import ClassificationDatasetBuilder
from model import ClassificationXLNet
from optimizer import create_optimizer
from model_runners import XLNetClassificationTrainer


flags.DEFINE_list(
    'input_file_paths', None, 'Paths to input TFRecord files.')
flags.DEFINE_string(
    'pretrain_model_path', None, 'Path to pretrained XLNet model.')
flags.DEFINE_integer(
    'batch_size', 4, 'Number of sequences in a batch.')
flags.DEFINE_integer(
    'seq_len', 512, 'Sequence length.')
flags.DEFINE_string(
    'spiece_model_path', None, 'Path to SentencePiece model.')
flags.DEFINE_integer(
    'stack_size', 12, 'Number of layers in Transformer Encoder.')
flags.DEFINE_integer(
    'hidden_size', 768, 'Hidden size.')
flags.DEFINE_integer(
    'num_heads', 12, 'Number of attention heads.')
flags.DEFINE_float(
    'dropout_rate', 0.1, 'Dropout rate.')
flags.DEFINE_float(
    'dropout_rate_attention', 0.1, 'Dropout rate attention.')
flags.DEFINE_bool(
    'tie_biases', False, 'Whether to tie biases.')

flags.DEFINE_float(
    'init_lr', 2e-5, 'Initial learning rate.')
flags.DEFINE_integer(
    'num_train_steps', 32000, 'Number of training iterations.')
flags.DEFINE_integer(
    'num_warmup_steps', 4000, 'Number of warm-up training iterations.')
flags.DEFINE_float(
    'min_lr_ratio', 0.0, 'The final learning rate will be'
        '`min_lr_ratio * init_lr.`')
flags.DEFINE_float(
    'adam_epsilon', 1e-8, 'The small values used in Adam optimizer.')

flags.DEFINE_string(
    'ckpt_path', 'models/imdb/', 'The path to the directory where the '
        'checkpoint files of the fine-tuned XLNet model will be written to.')
flags.DEFINE_integer(
    'persist_per_iterations', 8000, 'Save checkpiont files every '
        '`persist_per_iterations` iterations.')
flags.DEFINE_integer(
    'log_per_iterations', 100, 'Prints log info every `log_per_iterations` '
        'iterations.')

FLAGS = flags.FLAGS


def main(_):
  batch_size = FLAGS.batch_size
  seq_len = FLAGS.seq_len
  input_file_paths = FLAGS.input_file_paths
  spiece_model_path = FLAGS.spiece_model_path
  stack_size = FLAGS.stack_size
  hidden_size = FLAGS.hidden_size
  num_heads = FLAGS.num_heads
  filter_size = hidden_size * 4
  dropout_rate = FLAGS.dropout_rate
  dropout_rate_attention = FLAGS.dropout_rate_attention
  tie_biases = FLAGS.tie_biases
  init_lr = FLAGS.init_lr
  num_train_steps = FLAGS.num_train_steps
  num_warmup_steps = FLAGS.num_warmup_steps
  min_lr_ratio = FLAGS.min_lr_ratio
  adam_epsilon = FLAGS.adam_epsilon
  ckpt_path = FLAGS.ckpt_path
  persist_per_iterations = FLAGS.persist_per_iterations
  log_per_iterations = FLAGS.log_per_iterations
  pretrain_model_path = FLAGS.pretrain_model_path

  sp = spm.SentencePieceProcessor()
  sp.Load(spiece_model_path)
  vocab_size = sp.vocab_size()

  # create classification XLNet model
  model = ClassificationXLNet(vocab_size=vocab_size,
                              stack_size=stack_size,
                              hidden_size=hidden_size,
                              num_heads=num_heads,
                              filter_size=filter_size,
                              dropout_rate=dropout_rate,
                              dropout_rate_attention=dropout_rate_attention,
                              tie_biases=tie_biases,
                              num_classes=2)

  # training dataset
  builder = ClassificationDatasetBuilder(batch_size, seq_len, True)
  dataset = builder.build_dataset(input_file_paths)

  # create optimizer
  optimizer = create_optimizer(init_lr,
                               num_train_steps,
                               num_warmup_steps,
                               min_lr_ratio,
                               adam_epsilon)
  # checkpoint
  ckpt = tf.train.Checkpoint(model=model)

  # create trainer and start training
  trainer = XLNetClassificationTrainer(pretrain_model_path,
                                       ckpt_path,
                                       num_train_steps,
                                       persist_per_iterations,
                                       log_per_iterations)

  trainer.train(model, dataset, optimizer, ckpt)


if __name__ == '__main__':
  flags.mark_flag_as_required('input_file_paths')
  flags.mark_flag_as_required('spiece_model_path')
  flags.mark_flag_as_required('pretrain_model_path')
  app.run(main)

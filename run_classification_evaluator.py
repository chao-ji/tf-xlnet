"""Evaluates an XLNet model purposed for sequence classification task on IMDB
dataset.
"""
import sentencepiece as spm
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

from commons.dataset import ClassificationDatasetBuilder
from model import ClassificationXLNet
from model_runners import evaluate_classification_xlnet


flags.DEFINE_integer(
    'batch_size', 16, 'Number of sequences in a batch.')
flags.DEFINE_integer(
    'seq_len', 512, 'Length of sequence in a batch.')
flags.DEFINE_list(
    'input_file_paths', None, 'Paths to the TFRecord files of evaluation data.')

flags.DEFINE_string(
    'spiece_model_path', None, 'Path to SentencePiece model.')
flags.DEFINE_integer(
    'stack_size', 12, 'Number of layers in Transformer Encoder.')
flags.DEFINE_integer(
    'hidden_size', 768, 'The hidden size of continuous representation.')
flags.DEFINE_integer(
    'num_heads', 12, 'Number of attention heads.')
flags.DEFINE_bool(
    'tie_biases', False, 'Whether to force all layers use the same content bias'
        ' and position bias (T), or create the biases for each layer (F).')
flags.DEFINE_string(
    'ckpt_path', None, 'The path to the directory where the checkpoint files of'
        ' the fine-tuned XLNet model will be loaded from.')

FLAGS = flags.FLAGS


def main(_):
  batch_size = FLAGS.batch_size
  seq_len = FLAGS.seq_len
  input_file_paths = FLAGS.input_file_paths
  stack_size = FLAGS.stack_size
  hidden_size = FLAGS.hidden_size
  filter_size = hidden_size * 4
  num_heads = FLAGS.num_heads
  tie_biases = FLAGS.tie_biases
  ckpt_path = FLAGS.ckpt_path
  spiece_model_path = FLAGS.spiece_model_path

  sp = spm.SentencePieceProcessor()
  sp.Load(spiece_model_path)
  vocab_size = sp.vocab_size()

  # create classification XLNet model
  model = ClassificationXLNet(vocab_size=vocab_size,
                              stack_size=stack_size,
                              hidden_size=hidden_size,
                              num_heads=num_heads,
                              filter_size=filter_size,
                              dropout_rate=0.0,
                              dropout_rate_attention=0.0,
                              tie_biases=tie_biases,
                              num_classes=2)

  # evaluation dataset
  builder = ClassificationDatasetBuilder(batch_size, seq_len)
  dataset = builder.build_dataset(input_file_paths)

  # restore model
  ckpt = tf.train.Checkpoint(model=model)
  ckpt.restore(ckpt_path)

  correct, total = evaluate_classification_xlnet(model, dataset)
  print('correct', correct, 'total', total)
  logging.info(f'Num of correct: {correct}')
  logging.info(f'Total num: {total}')
  logging.info(f'Accuracy: {correct / total}')

if __name__ == '__main__':
  flags.mark_flag_as_required('input_file_paths')
  flags.mark_flag_as_required('spiece_model_path')
  flags.mark_flag_as_required('ckpt_path')
  app.run(main)

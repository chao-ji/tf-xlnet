"""
"""
import sentencepiece as spm
import tensorflow as tf
from absl import app
from absl import flags

from commons.dataset import ClassificationDatasetBuilder
from model import ClassificationXLNet
from model_runners import XLNetClassificationEvaluator


flags.DEFINE_integer(
    'batch_size', 16, 'Number of sequences in a batch.')
flags.DEFINE_integer(
    'seq_len', 512, 'Sequence length.')
flags.DEFINE_list(
    'input_file_paths', None, 'Paths to input TFRecord files.')

flags.DEFINE_string(
    'spm_path', None, 'Path to SentencePiece model.')
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
flags.DEFINE_string(
    'ckpt_path', None, 'Path to checkpoint file.')

FLAGS = flags.FLAGS


def main(_):

  batch_size = FLAGS.batch_size
  seq_len = FLAGS.seq_len

  input_file_paths = FLAGS.input_file_paths

  builder = ClassificationDatasetBuilder(batch_size, seq_len)
  dataset = builder.build_dataset(input_file_paths)

  stack_size = FLAGS.stack_size
  hidden_size = FLAGS.hidden_size
  filter_size = hidden_size * 4
  num_heads = FLAGS.num_heads
  tie_biases = FLAGS.tie_biases


  spm_path = FLAGS.spm_path
  sp = spm.SentencePieceProcessor()
  sp.LoadFromFile(spm_path)
  vocab_size = sp.vocab_size()


  model = ClassificationXLNet(vocab_size=vocab_size,
                       stack_size=stack_size,
                       hidden_size=hidden_size,
                       num_heads=num_heads,
                       filter_size=filter_size,
                       dropout_rate=0.0,
                       dropout_rate_attention=0.0,
                       tie_biases=tie_biases,
                       num_classes=2)

  ckpt_path = FLAGS.ckpt_path
  ckpt = tf.train.Checkpoint(model=model)
  ckpt.restore(ckpt_path)

  evaluator = XLNetClassificationEvaluator(model)

  correct, total = evaluator.evaluate(dataset)
  print('correct', correct, 'total', total)  

 
if __name__ == '__main__':
  flags.mark_flag_as_required('input_file_paths')
  flags.mark_flag_as_required('spm_path')
  flags.mark_flag_as_required('ckpt_path')
  app.run(main)
 

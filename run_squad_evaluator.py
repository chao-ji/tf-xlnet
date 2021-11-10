""""""
import json
import pickle

import sentencepiece as spm
import tensorflow as tf
from absl import app
from absl import flags

from commons.dataset import SquadDatasetBuilder
from model import QuestionAnswerXLNet
from model_runners import XLNetQuestionAnswerEvaluator


flags.DEFINE_integer(
    'batch_size', 16, 'Number of sequences in a batch.')
flags.DEFINE_integer(
    'seq_len', 512, 'Sequence length.')
flags.DEFINE_list(
    'input_file_paths', None, 'Paths to input TFRecord files.')
flags.DEFINE_string(
    'orig_file_path', None, 'Path to the original json file of the dev split '
    'of SQuAD 2.0 dataset.')
flags.DEFINE_string(
    'feature_file_path', None, 'Path to the file containing sequence features '
    'of dev split.')
flags.DEFINE_string(
    'spm_path', None, 'Path to SentencePiece model.')
flags.DEFINE_integer(
    'start_n_top', 5, 'Beam size of the number of start locations.')
flags.DEFINE_integer(
    'end_n_top', 5, 'Beam size of the number of end locations.')
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
flags.DEFINE_integer(
    'n_best_size', 5, 'n best size')
flags.DEFINE_integer(
    'max_answer_length', 64, 'max answer length')
flags.DEFINE_string(
    'predict_dir', '/home/chaoji/Downloads/models-2.4.0/official/nlp/xlnet/model/squad/predictions', 'predict dir')

FLAGS = flags.FLAGS


def main(_):
  
  batch_size = FLAGS.batch_size
  seq_len = FLAGS.seq_len

  builder = SquadDatasetBuilder(batch_size, seq_len)

  input_file_paths = FLAGS.input_file_paths

  dataset = builder.build_dataset(input_file_paths)

  orig_file_path = FLAGS.orig_file_path

  start_n_top = FLAGS.start_n_top
  end_n_top = FLAGS.end_n_top


  n_best_size = FLAGS.n_best_size
  max_answer_length = FLAGS.max_answer_length
  predict_dir = FLAGS.predict_dir

  with open('dev.pickle', 'rb') as f:
      eval_examples = pickle.load(f)


  feature_file_path = FLAGS.feature_file_path
  with tf.io.gfile.GFile(feature_file_path, 'rb') as f:
      eval_features = pickle.load(f)

  with tf.io.gfile.GFile(orig_file_path) as f:
      original_data = json.load(f)["data"]


  spm_path = FLAGS.spm_path
  sp = spm.SentencePieceProcessor()
  sp.LoadFromFile(spm_path)
  vocab_size = sp.vocab_size()

  stack_size = FLAGS.stack_size
  hidden_size = FLAGS.hidden_size
  filter_size = hidden_size * 4
  num_heads = FLAGS.num_heads
  tie_biases = FLAGS.tie_biases

  model = QuestionAnswerXLNet(vocab_size=vocab_size,
                              stack_size=stack_size,
                              hidden_size=hidden_size,
                              num_heads=num_heads,
                              filter_size=filter_size,
                              dropout_rate=0.0,
                              dropout_rate_attention=0.0,
                              tie_biases=tie_biases,
                              start_n_top=start_n_top,
                              end_n_top=end_n_top)

  ckpt = tf.train.Checkpoint(model=model)

  ckpt_path = FLAGS.ckpt_path
  ckpt.restore(ckpt_path)

  evaluator = XLNetQuestionAnswerEvaluator(model)

  out = evaluator.evaluate(dataset,
                           eval_examples,
                           eval_features,
                           original_data, 
                           start_n_top,
                           end_n_top,
                           n_best_size,
                           max_answer_length,
                           predict_dir)
  print(out)


if __name__ == '__main__':
  flags.mark_flag_as_required('input_file_paths')
  flags.mark_flag_as_required('spm_path')
  flags.mark_flag_as_required('orig_file_path')
  flags.mark_flag_as_required('feature_file_path')
  flags.mark_flag_as_required('ckpt_path')
  app.run(main)
 

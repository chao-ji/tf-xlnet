"""Evaluates an XLNet model purposed for question answering task on SQuAD
dataset.
"""
import json
import pickle

import sentencepiece as spm
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

from commons.dataset import SquadDatasetBuilder
from model import QuestionAnswerXLNet
from model_runners import evaluate_squad_xlnet


flags.DEFINE_integer(
    'batch_size', 16, 'Number of sequences in a batch.')
flags.DEFINE_integer(
    'seq_len', 512, 'Length of sequence in a batch.')
flags.DEFINE_list(
    'input_file_paths', None, 'Paths to the TFRecord files of evaluation data.')
flags.DEFINE_string(
    'orig_file_path', None, 'Path to the original json file of the dev split '
        'of SQuAD 2.0 dataset.')
flags.DEFINE_string(
    'feature_file_path', None, 'Path to the file containing sequence features '
        'of dev split.')
flags.DEFINE_string(
    'spiece_model_path', None, 'Path to SentencePiece model.')
flags.DEFINE_integer(
    'start_n_top', 5, 'Beam size of the number of start locations.')
flags.DEFINE_integer(
    'end_n_top', 5, 'Beam size of the number of end locations.')
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
flags.DEFINE_integer(
    'n_best_size', 5, 'Bumber of best scoring predictions.')
flags.DEFINE_integer(
    'max_answer_length', 64, 'max length of answer text.')
flags.DEFINE_string(
    'predict_dir', 'models/squad/predictions', 'Path to the directory where '
        'prediction files will be written to.')

FLAGS = flags.FLAGS


def main(_):
  batch_size = FLAGS.batch_size
  seq_len = FLAGS.seq_len
  input_file_paths = FLAGS.input_file_paths
  orig_file_path = FLAGS.orig_file_path
  start_n_top = FLAGS.start_n_top
  end_n_top = FLAGS.end_n_top
  n_best_size = FLAGS.n_best_size
  max_answer_length = FLAGS.max_answer_length
  predict_dir = FLAGS.predict_dir
  spiece_model_path = FLAGS.spiece_model_path
  feature_file_path = FLAGS.feature_file_path
  ckpt_path = FLAGS.ckpt_path
  stack_size = FLAGS.stack_size
  hidden_size = FLAGS.hidden_size
  filter_size = hidden_size * 4
  num_heads = FLAGS.num_heads
  tie_biases = FLAGS.tie_biases

  sp = spm.SentencePieceProcessor()
  sp.Load(spiece_model_path)
  vocab_size = sp.vocab_size()

  # create question answer XLNet model
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

  # evaluation dataset
  builder = SquadDatasetBuilder(batch_size, seq_len)
  dataset = builder.build_dataset(input_file_paths)

  with tf.io.gfile.GFile(feature_file_path, 'rb') as f:
    eval_feature_list = pickle.load(f)
  with tf.io.gfile.GFile(orig_file_path) as f:
    orig_data = json.load(f)["data"]

  # restore model
  ckpt = tf.train.Checkpoint(model=model)
  ckpt.restore(ckpt_path)

  # run evaluation
  results = evaluate_squad_xlnet(model,
                                 dataset,
                                 eval_feature_list,
                                 orig_data,
                                 start_n_top,
                                 end_n_top,
                                 n_best_size,
                                 max_answer_length,
                                 predict_dir)

  logging.info(f'best_exact: {results["best_exact"]:9.6f}')
  logging.info(f'best_exact_thres: {results["best_exact_thresh"]:9.6f}')
  logging.info(f'best_f1: {results["best_f1"]:9.6f}')
  logging.info(f'best_f1_thresh: {results["best_f1_thresh"]:9.6f}')
  logging.info(f'has_ans_exact: {results["has_ans_exact"]:9.6f}')
  logging.info(f'has_ans_f1: {results["has_ans_f1"]:9.6f}')


if __name__ == '__main__':
  flags.mark_flag_as_required('input_file_paths')
  flags.mark_flag_as_required('spiece_model_path')
  flags.mark_flag_as_required('orig_file_path')
  flags.mark_flag_as_required('feature_file_path')
  flags.mark_flag_as_required('ckpt_path')
  app.run(main)

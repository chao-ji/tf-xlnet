"""Builds & runs an XLNet model purposed for pretraining on permutation
language modeling task.
"""
import sentencepiece as spm
import tensorflow as tf
from absl import app
from absl import flags

from commons.dataset import XLNetPretrainDatasetBuilder
from model import PretrainingXLNet
from model_runners import XLNetPretrainer
from optimizer import create_optimizer
from text_utils import CLS_ID
from text_utils import SEP_ID


flags.DEFINE_list(
    'input_file_paths', None, 'Paths to the TFRecord files of training data.')
flags.DEFINE_integer(
    'batch_size', 2, 'Number of sequences in a batch.')
flags.DEFINE_string(
    'spiece_model_path', None, 'Path to SentencePiece model file.')
flags.DEFINE_integer(
    'seq_len', 512, 'Length of sequence in a batch.')
flags.DEFINE_integer(
    'reuse_len', 256, 'Number of token that can be reused as memory.')
flags.DEFINE_integer(
    'perm_size', 256, 'Window size of the permutation.')
flags.DEFINE_float(
    'leak_ratio', 0.1, 'Percent of masked tokens that are can be attended to.')
flags.DEFINE_integer(
    'num_predict', 85, 'Number of tokens to predict.')
flags.DEFINE_integer(
    'max_num_tokens', 5, 'Maximum number of tokens to sample in a span.')
flags.DEFINE_integer(
    'min_num_tokens', 1, 'Minimum number of tokens to sample in a span.')

flags.DEFINE_float(
    'init_lr', 1e-4, 'Initial learning rate.')
flags.DEFINE_integer(
    'num_train_steps', 2191224, 'Number of training iterations.')
flags.DEFINE_integer(
    'num_warmup_steps', 180000, 'Number of warm-up training iterations.')
flags.DEFINE_float(
    'min_lr_ratio', 0.0, 'The final learning rate will be'
        '`min_lr_ratio * init_lr.`')
flags.DEFINE_float(
    'adam_epsilon', 1e-8, 'The small values used in Adam optimizer.')

flags.DEFINE_integer(
    'stack_size', 12, 'Number of layers in Transformer Encoder.')
flags.DEFINE_integer(
    'hidden_size', 768, 'The hidden size of continuous representation.')
flags.DEFINE_integer(
    'num_heads', 12, 'Number of attention heads.')

flags.DEFINE_integer(
    'mem_len', 384, 'Number of tokens to be cached.')
flags.DEFINE_float(
    'dropout_rate', 0.1, 'Dropout rate for dropout layers')
flags.DEFINE_float(
    'dropout_rate_attention', 0.1, 'Dropout rate applied on the '
        'query-to-reference attention matrix.')
flags.DEFINE_bool(
    'tie_biases', False, 'Whether to force all layers use the same content bias'
        ' and position bias (T), or create the biases for each layer (F).')

flags.DEFINE_string(
    'output_dir', 'models/pretrain', 'Directory where checkpoint files will be '
        'saved.')
flags.DEFINE_integer(
    'persist_per_iterations', 12000, 'Save checkpiont files every '
    '`persist_per_iterations` iterations.')
flags.DEFINE_integer(
    'log_per_iterations', 100, 'Prints log info every `log_per_iterations` '
    'iterations.')

FLAGS = flags.FLAGS


def main(_):
  seq_len = FLAGS.seq_len
  reuse_len = FLAGS.reuse_len
  batch_size = FLAGS.batch_size
  num_predict = FLAGS.num_predict
  perm_size = FLAGS.perm_size
  leak_ratio = FLAGS.leak_ratio
  max_num_tokens = FLAGS.max_num_tokens
  min_num_tokens = FLAGS.min_num_tokens
  input_file_paths = FLAGS.input_file_paths
  init_lr = FLAGS.init_lr
  num_train_steps = FLAGS.num_train_steps
  num_warmup_steps = FLAGS.num_warmup_steps
  min_lr_ratio = FLAGS.min_lr_ratio
  adam_epsilon = FLAGS.adam_epsilon
  spiece_model_path = FLAGS.spiece_model_path
  stack_size = FLAGS.stack_size
  hidden_size = FLAGS.hidden_size
  num_heads = FLAGS.num_heads
  filter_size = hidden_size * 4
  mem_len = FLAGS.mem_len
  dropout_rate = FLAGS.dropout_rate
  dropout_rate_attention = FLAGS.dropout_rate_attention
  tie_biases = FLAGS.tie_biases
  output_dir = FLAGS.output_dir
  persist_per_iterations = FLAGS.persist_per_iterations
  log_per_iterations = FLAGS.log_per_iterations

  sp = spm.SentencePieceProcessor()
  sp.Load(spiece_model_path)
  vocab_size = sp.vocab_size()

  # create pretraining XLNet model
  model = PretrainingXLNet(vocab_size=vocab_size,
                           stack_size=stack_size,
                           hidden_size=hidden_size,
                           num_heads=num_heads,
                           filter_size=filter_size,
                           mem_len=mem_len,
                           reuse_len=reuse_len,
                           dropout_rate=dropout_rate,
                           dropout_rate_attention=dropout_rate_attention,
                           tie_biases=tie_biases)

  # traininig dataset
  builder = XLNetPretrainDatasetBuilder(seq_len,
                                        reuse_len,
                                        batch_size,
                                        num_predict,
                                        perm_size,
                                        leak_ratio,
                                        max_num_tokens,
                                        min_num_tokens,
                                        CLS_ID,
                                        SEP_ID)
  dataset = builder.build_dataset(input_file_paths)

  # create optimizer
  optimizer = create_optimizer(init_lr,
                               num_train_steps,
                               num_warmup_steps,
                               min_lr_ratio,
                               adam_epsilon)

  # checkpoint
  ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

  # create trainer and start training
  trainer = XLNetPretrainer(output_dir,
                            num_train_steps,
                            persist_per_iterations,
                            log_per_iterations)
  trainer.train(model, dataset, optimizer, ckpt)


if __name__ == '__main__':
  flags.mark_flag_as_required('input_file_paths')
  app.run(main)

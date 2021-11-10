""""""
import sentencepiece as spm
import tensorflow as tf
from absl import app
from absl import flags

from commons.dataset import SquadDatasetBuilder
from model import QuestionAnswerXLNet
from model_runners import XLNetQuestionAnswerTrainer
from optimizer import create_optimizer


flags.DEFINE_list(
    'input_file_paths', None, 'Paths to input TFRecord files.')
flags.DEFINE_string(
    'pretrain_model_path', None, 'Path to pretrained XLNet model.')
flags.DEFINE_integer(
    'batch_size', 4, 'Number of sequences in a batch.')
flags.DEFINE_integer(
    'seq_len', 512, 'Sequence length.')
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

flags.DEFINE_float(
    'init_lr', 3e-5, 'Initial learning rate.')
flags.DEFINE_integer(
    'num_train_steps', 96000, 'Number of training iterations.')
flags.DEFINE_integer(
    'num_warmup_steps', 12000, 'Number of warm-up training iterations.')
flags.DEFINE_float(
    'min_lr_ratio', 0.0, 'Minimum learning rate ratio.')
flags.DEFINE_float(
    'adam_epsilon', 1e-6, 'Adam epsilon.')
flags.DEFINE_float(
    'weight_decay_rate', 0.01, 'Weight decay rate.')

flags.DEFINE_float(
    'lr_layer_decay_rate', 0.75, 'lr layer decay rate')
flags.DEFINE_string(
    'ckpt_path', '.', 'ckpt path')
flags.DEFINE_integer(
    'persist_per_iterations', 12000, 'persist per iterations') 
flags.DEFINE_integer(
    'log_per_iterations', 1000, 'log per iterations')

FLAGS = flags.FLAGS


def main(_):
  batch_size = FLAGS.batch_size
  seq_len = FLAGS.seq_len
  input_file_paths = FLAGS.input_file_paths 

  builder = SquadDatasetBuilder(batch_size, seq_len, True)
  dataset = builder.build_dataset(input_file_paths)

  spm_path = FLAGS.spm_path
  sp = spm.SentencePieceProcessor()
  sp.LoadFromFile(spm_path)
  vocab_size = sp.vocab_size()
  stack_size = FLAGS.stack_size
  hidden_size = FLAGS.hidden_size
  num_heads = FLAGS.num_heads
  filter_size = hidden_size * 4
  dropout_rate = FLAGS.dropout_rate
  dropout_rate_attention = FLAGS.dropout_rate_attention
  tie_biases = FLAGS.tie_biases


  model = QuestionAnswerXLNet(vocab_size=vocab_size,
                              stack_size=stack_size,
                              hidden_size=hidden_size,
                              num_heads=num_heads,
                              filter_size=filter_size,
                              dropout_rate=dropout_rate,
                              dropout_rate_attention=dropout_rate_attention,
                              tie_biases=tie_biases,
                              start_n_top=5,
                              end_n_top=5)

  pretrain_model_path = FLAGS.pretrain_model_path

  ckpt = tf.train.Checkpoint(model=model)
  ckpt.restore(pretrain_model_path).expect_partial()
  print('Loaded checkpoint %s' % pretrain_model_path)
    
  init_lr = FLAGS.init_lr
  num_train_steps = FLAGS.num_train_steps
  num_warmup_steps = FLAGS.num_warmup_steps
  min_lr_ratio = FLAGS.min_lr_ratio
  adam_epsilon = FLAGS.adam_epsilon
  weight_decay_rate = FLAGS.weight_decay_rate

  optimizer = create_optimizer(init_lr,
                               num_train_steps,
                               num_warmup_steps,
                               min_lr_ratio,
                               weight_decay_rate,
                               adam_epsilon)

  lr_layer_decay_rate = FLAGS.lr_layer_decay_rate
  ckpt_path = FLAGS.ckpt_path
  persist_per_iterations = FLAGS.persist_per_iterations
  log_per_iterations = FLAGS.log_per_iterations 

  trainer = XLNetQuestionAnswerTrainer(model, lr_layer_decay_rate)
  trainer.train(dataset, optimizer, ckpt, ckpt_path, num_train_steps, persist_per_iterations, log_per_iterations)


if __name__ == '__main__':
  flags.mark_flag_as_required('input_file_paths')
  flags.mark_flag_as_required('spm_path')
  flags.mark_flag_as_required('pretrain_model_path')
  app.run(main)



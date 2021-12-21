"""Defines functions and contants for text processing."""
import re
import string
import unicodedata

SPIECE_UNDERLINE = '▁'
SEG_ID_P = 0
SEG_ID_Q = 1
SEG_ID_CLS = 2
SEG_ID_PAD = 3
CLS_ID = 3
SEP_ID = 4
EOD_ID = 7


def normalize_text(text, lower=False, remove_space=True, keep_accents=False):
  """Normalizes input text by optionally removing redundant whitespaces,
  dropping accents, and lowering cases.

  Args:
    text: string scalar, input text (unicode string).
    lower: (Optional) bool scalar, whether to lower-case text.
    remove_space: (Optional) bool scalar, whether to remove whitespaces.
    keep_accents: (Optional) bool scalar, whether to keep accents.

  Returns:
    text: string scalar, output text (unicode string).
  """
  if remove_space:
    text = ' '.join(text.strip().split())

  text = text.replace('``', '"').replace("''", '"')

  if not keep_accents:
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
  if lower:
    text = text.lower()

  return text


def normalize_answer(text):
  """Normalize answer text by lowercasing and removing punctuation, articles and
  extra whitespace.

  Args:
    text: string scalar, input text.

  Returns:
    texst: string scalar, normalized texst.
  """
  text = text.lower()
  # remove puntuation chars
  exclude = set(string.punctuation)
  text = ''.join(ch for ch in text if ch not in exclude)
  # remove articles
  regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
  text = re.sub(regex, ' ', text)
  # sperate by whitespace
  text = ' '.join(text.split())
  return text


def preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
  """Preprocesses texts.

  Args:
    inputs: string scalar, input text.
    lower: (Optional) bool scalar, whether to lowercase text. Defaults to False
    remove_space: (Optional) bool scalar, whether to remove leading and trailing
      whitespaces. Defaults to True.
    keep_accents: (Optional) bool scalar, whether to keep accents of accented
      letters.

  Returns:
    outputs: string scalar, processed text.
  """
  if remove_space:
    outputs = ' '.join(inputs.strip().split())
  else:
    outputs = inputs

  outputs = outputs.replace('``', '"').replace("''", '"')

  if not keep_accents:
    outputs = unicodedata.normalize('NFKD', outputs)
    outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
  if lower:
    outputs = outputs.lower()

  return outputs


def encode_ids(sp_model, text):
  """Split up text into subword tokens and convert to token IDs.

  Args:
    sp_model: instance of SentencePieceProcessor, sentence piece model.
    text: string scalar, input text (unicode string).

  Returns:
    ids: list of integers, token ids.
  """
  pieces = encode_pieces(sp_model, text)
  ids = [sp_model.PieceToId(piece) for piece in pieces]
  return ids


def encode_pieces(sp_model, text):
  """Split up text into subword tokens.

  Args:
    sp_model: instance of SentencePieceProcessor, sentence piece model.
    text: string scalar, input text (unicode string).

  Return:
    new_pieces: list of strings, subword tokens.
  """
  pieces = sp_model.EncodeAsPieces(text)
  new_pieces = []
  for piece in pieces:
    if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
      cur_pieces = sp_model.EncodeAsPieces(piece[:-1].replace(
          SPIECE_UNDERLINE, ''))
      if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
        if len(cur_pieces[0]) == 1:
          cur_pieces = cur_pieces[1:]
        else:
          cur_pieces[0] = cur_pieces[0][1:]
      cur_pieces.append(piece[-1])
      new_pieces.extend(cur_pieces)
    else:
      new_pieces.append(piece)

  return new_pieces

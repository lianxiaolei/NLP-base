"""

"""

import numpy as np
import tensorflow as tf
from hbconfig import Config


def print_variables(variables, rev_vocab=None, every_n_iter=100):
  """
  Prints the given tensors every N local steps, every N seconds, or at end.
  Args:
    variables:
    rev_vocab:
    every_n_iter:

  Returns:

  """
  values = tf.train.LoggingTensorHook(variables, every_n_iter,
                                      formatter=format_variable(variables,
                                                                rev_vocab=rev_vocab))
  return values


def format_variable(keys, rev_vocab=None):
  """
  Covert the word index array to word index or word sentences.
  Args:
    keys:
    rev_vocab:

  Returns:

  """

  def to_str(sequence):
    if type(sequence) == np.array:
      tokens = [rev_vocab.get(key, '') for key in sequence if key != Config.data.PAD_ID]
      return ' '.join(tokens)
    else:
      key = int(sequence)
      return rev_vocab.get(key)

  def format(values):
    result = []
    for key in keys:
      if rev_vocab is None:
        result.append(f"{key} = {values[key]}")
      else:
        result.append(f"{key} = {to_str(values[key])}")

    try:
      return '\n - '.join(result)
    except:
      pass

  return format

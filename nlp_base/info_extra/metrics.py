# coding:utf8

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import traceback

def gen_metrics(sequence_len, preds, labels, max_len, batch_size):
  """

  Args:
    sequence_len:
    binary_pred:
    pred:
    target_correctness:

  Returns:

  """
  sparse_preds = []
  sparse_labels = []
  # preds = np.reshape(preds, [BATCH_SIZE, max_len, 2])
  # preds = np.reshape(preds, [batch_size, max_len])
  for seq_idx, seq_len in enumerate(sequence_len):
    sparse_preds.append(np.argmax(preds[seq_idx, :seq_len, :], axis=-1))
    # sparse_preds.append(np.where(preds[seq_idx, :seq_len] > 0.5, 1, 0))
    sparse_labels.append(labels[seq_idx, :seq_len])

  # new_binary_pred = np.concatenate(binary_preds)
  # print('sparse_preds', sparse_preds)
  sparse_preds = np.concatenate(sparse_preds)
  sparse_labels = np.concatenate(sparse_labels)

  result = []
  for i in range(sparse_labels.shape[0]):
    result.append(1) if sparse_labels[i] == sparse_preds[i] else result.append(0)
  auc = 0
  accuracy = sum(result) / len(result)

  return auc, accuracy

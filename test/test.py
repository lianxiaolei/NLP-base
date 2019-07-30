# coding:utf8

def get_w2i_map(fname, preserve_zero=True):
  """
  Generate word set.
  Args:
    fname:         Source file name.
    preserve_zero: Preserve the place for '<UNK>'.

  Returns:
    word_set:      A set contains word index.

  """
  word_list = []

  with open(fname, 'r', encoding='utf8') as fin:
    content = fin.read().replace('\n\n', '')
    content_list = content.split('\n')
    for line in content_list:
      for phrase in line.split(' '):
        if len(phrase) < 3: continue
        if not '/' in phrase:
          words = phrase
        else:
          words, tag = phrase.split('/')
        words = words.split('_')
        word_list.extend(words)
  word_set = set(word_list)
  word_set = sorted(word_set)

  if preserve_zero:
    word_set.insert(0, '<UNK>')

  return word_set


training_word = get_w2i_map('/home/lian/data/nlp/datagrand_info_extra/train.txt')
test_word = get_w2i_map('/home/lian/data/nlp/datagrand_info_extra/test.txt')
print(len(training_word), training_word)
print(len(test_word), test_word)

none_co = []
for word in training_word:
  if word not in test_word:
    none_co.append(word)
print('training set have and test set not have', len(none_co), none_co)
del none_co
none_co = []
for word in test_word:
  if word not in training_word:
    none_co.append(word)
print('test set have and training set not have', len(none_co), none_co)

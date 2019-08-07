# coding:utf8

"""
Common tools for this project.
"""

import functools
import itertools
import operator


def functools_reduce(a):
  print('Reducing...')
  return functools.reduce(operator.concat, a)


def itertools_chain(a):
  return list(itertools.chain.from_iterable(a))


def reduce(tables):
  print('Reducing...')
  result = []
  for table in tables:
    while ' ' in table:
      table.remove(' ')
    result += table
  return result

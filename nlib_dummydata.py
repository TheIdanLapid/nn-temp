import numpy as np
from random import shuffle, choice

def dots(num, scale=1):
    return [[np.random.random() * scale for i in range(2)] for n in range(num)]


def line_dot(axis, dot_num, incline, const):
    def line_f(x): return x * incline + const
    samples = dots(axis, num)
    shuffle(set)
    set = [[{'data': s, 'label': {True: 1, False: -1}}[line_f(s) > s]] for s in samples]
    return set


def xor_data():
    series = [{'data': [0, 1], 'label': [1]},
              {'data': [1, 0], 'label': [1]},
              {'data': [0, 0], 'label': [-1]},
              {'data': [1, 1], 'label': [-1]}
              ]

    item = choice(series)
    # items = np.array([s['data'] for s in set])
    # labels = np.array([s['label'] for s in set])

    return item


def noise(dim, num, scale=1):
    return [[np.random.random() * scale for d in range(dim)] for n in range(num)]

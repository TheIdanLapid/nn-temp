import numpy as np

def sig(x): return 1 / (1 + np.exp(-x))


def dx_sig(s): return s * (1 - s)


def splus(x): return 1 + (np.exp(x))


def dx_splus(): sig


def relu(x): return max(0, x)


def dx_relu(r): return {True: 0, False: 1}[r == 0]

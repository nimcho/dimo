"""
Weightings
==========

Use if you have a raw co-occurrence matrix
in scipy.sparse.csr_matrix data structure
(see the subsection 1.4.2)
"""

import numpy as np


def raising(m, coeff=0.75):
    m.data **= coeff


def log(m):
    m.data = np.log(1 + m.data)


def ppmi(m):

    all_sum = m.sum()
    row_sums = np.array(m.sum(axis=1))[:, 0]
    col_sums = np.array(m.sum(axis=0))[0, :]

    m.data *= all_sum

    denom = col_sums[m.indices]
    #
    for i in range(m.shape[0]):
        beg, end = m.indptr[i], m.indptr[i + 1]
        denom[beg:end] *= row_sums[i]
    #
    m.data /= 0.00001 + denom

    m.data = np.log(m.data).clip(min=0.0)


def log_dice(m):

    row_sums = np.array(m.sum(axis=1))[:, 0]
    col_sums = np.array(m.sum(axis=0))[0, :]

    m.data *= 2

    denom = col_sums[m.indices] + 0.00001
    #
    for i in range(m.shape[0]):
        beg, end = m.indptr[i], m.indptr[i + 1]
        denom[beg:end] += row_sums[i]
    #
    m.data /= denom

    m.data = (14 + np.log2(m.data)).clip(min=0.0)

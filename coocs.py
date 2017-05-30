#!/usr/bin/python
"""
Word-Word Co-Occurrence Matrix Builder
======================================

If you have a corpus in text file (one line -- one sentence),
run this from terminal (see `main` function)

... or for corpora in arbitrary format, use `count_coocs` function.
"""

import pickle
import sys
import numpy as np

from collections import defaultdict
from scipy.sparse import coo_matrix, csr_matrix

# Local imports
from misc import LineCorpus


REPORT_DELAY = 10**5  # in struct. items (e.g. sentences)
ARR_SIZE = 10 ** 8


def count_coocs(corpus, output_name, min_count=1, window=4):
    """
    `corpus` should be a stream of sentences,
    where sentence is a non-empty list of words
    """

    assert min_count >= 1
    assert window >= 1

    print("Building vocab...")

    vocab = defaultdict(lambda: 0)

    for sentence in corpus:
        for word in sentence:
            vocab[word] += 1

    vocab = {w: c for w, c in vocab.items() if c >= min_count}

    print("Vocabulary built.")

    sorted_vocab = sorted(vocab.items(), key=lambda (w, c): c, reverse=True)
    word2i = {word: i for i, (word, _) in enumerate(sorted_vocab)}

    rows = np.zeros((ARR_SIZE, ), dtype=np.int32)
    cols = np.zeros((ARR_SIZE, ), dtype=np.int32)
    vals = np.zeros((ARR_SIZE, ), dtype=np.int32)
    ind = 0
    max_ind = ARR_SIZE
    shape = (len(vocab), len(vocab))

    m = coo_matrix(shape, dtype=np.float64)
    weights = np.array([(window - k) / window for k in range(window)])

    for i, sentence in enumerate(corpus):
        sentence = np.array([word2i[w] for w in sentence if w in word2i])

        for j in range(len(sentence)):
            target_id = sentence[j]

            for k in range(window):

                q = j + 1 + k
                if q >= len(sentence):
                    break
                context_id = sentence[q]

                if ind >= max_ind:
                    m = coo_matrix(
                        (np.concatenate([m.data, vals]),
                         (np.concatenate([m.row, rows]),
                          np.concatenate([m.col, cols]))),
                        shape=shape, dtype=np.float64,
                    )
                    m.sum_duplicates()
                    ind = 0

                rows[ind] = target_id
                cols[ind] = context_id
                vals[ind] = weights[k]
                ind += 1

        if i % REPORT_DELAY == 0:
            print("Sentence #%i" % i)

    m = coo_matrix(
        (np.concatenate([m.data, vals[:ind]]),
         (np.concatenate([m.row, rows[:ind]]),
          np.concatenate([m.col, cols[:ind]]))),
        shape=shape, dtype=np.float64
    )
    m.sum_duplicates()

    print("Counting completed.")

    # Make the context window and the matrix symmetric
    m_csr = csr_matrix(m)
    m = coo_matrix(m_csr + m_csr.transpose())

    np.save(output_name + "-rows.npy", m.row)
    np.save(output_name + "-cols.npy", m.col)
    np.save(output_name + "-vals.npy", m.data)

    with open(output_name + "-target2i.pickle", "w") as f:
        pickle.dump(word2i, f)


def main():

    if len(sys.argv) != 5:
        sys.stderr.write("Usage: python coocs.py "
                         "CORPUS_FILE OUTPUT_NAME MIN_COUNT WINDOW_SIZE\n")
        sys.exit(1)

    corpus_file = sys.argv[1]
    output_name = sys.argv[2]
    min_count = int(sys.argv[3])
    window_size = int(sys.argv[4])

    corpus = LineCorpus(corpus_file)

    count_coocs(corpus, output_name, min_count, window_size)


if __name__ == "__main__":
    main()

#!/usr/bin/python
#
# This code is not optimized in any special way
# aside from what was really necessary.

import pickle
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import KeyedVectors

# Local imports
from deval import DiMo


class SkEThes(DiMo):
    """
    Abstract class for SkEThes
        ... or basically for an arbitrary target-context matrix

    Subclasses should implement `similarity` and `similarities` methods.
    """

    def __init__(self, name, weighting=None):
        """
        Loads a target-context matrix defined by three files:
            [NAME]-rows.npy  # row indices
            [NAME]-cols.npy  # col indices
            [NAME]-vals.npy  # value on corresponding cell

        as well as a dictionary:
            [NAME]-target2i.pickle

        If the matrix contains raw counts, you may consider applying
        some weightings on it (see `weightings.py` file).
        """

        self.name = name

        with open(name + "-target2i.pickle") as f:
            word2i = pickle.load(f)

        i2word = {i: word for word, i in word2i.items()}

        super(SkEThes, self).__init__(word2i, i2word)

        rows = np.load(name + "-rows.npy")
        cols = np.load(name + "-cols.npy")
        scores = np.load(name + "-vals.npy")

        self.M = csr_matrix((scores, (rows, cols)))

        if weighting is not None:
            weighting(self.M)


class SkEThesCOS(SkEThes):
    """
    SkEThes using cosine similarity.
    """

    def __init__(self, name, *args, **kwargs):
        super(SkEThesCOS, self).__init__(name, *args, **kwargs)
        normalize(self.M, norm="l2", axis=1, copy=False)

    def similarity(self, a, b):
        i = (a if type(a) is int else self.word2i[a])
        j = (b if type(b) is int else self.word2i[b])
        return self.M[i, :].dot(self.M[j, :].transpose())[0, 0]

    def similarities(self, word):
        i = (word if type(word) is int else self.word2i[word])
        return self.M.dot(self.M[i, :].transpose()).toarray()[:, 0]


class SkEThesSKE(SkEThes):
    """
    SkEThes implementing the default similarity measure used in Sketch Engine.
    """

    def __init__(self, name, *args, **kwargs):

        super(SkEThesSKE, self).__init__(name, *args, **kwargs)

        # Some pre-computation:
        self.signs = self.M.sign()
        self.sums = self.M.sum(axis=1)

    def similarity(self, a, b):

        i = (a if type(a) is int else self.word2i[a])
        j = (b if type(b) is int else self.word2i[b])

        heu = (self.M[i, :] - self.M[j, :]).power(2) / 50
        upper_raw = self.M[i, :] + self.M[j, :] - heu

        # We want only those cells that are non-zero in both, i and j
        comb = self.signs[i, :].multiply(self.signs[j, :])
        upper = comb.multiply(upper_raw)

        res = upper.sum(axis=1) / (self.sums[i] + self.sums[j])
        return res[0, 0]

    def similarities(self, word):

        i = (word if type(word) is int else self.word2i[word])

        # Copy of M where each non-zero cell M[x, j]
        # is zeroed and removed if M[i, j] == 0
        Mnz = self.M.multiply(self.signs[i, :])
        Mnz.eliminate_zeros()

        # Copy of Mnz where each non-zero cell Mnz[x, j]
        # is equaled to Mnz[i, j].
        Mi = (Mnz != 0).multiply(self.M[i, :])

        inn = Mi + Mnz - ((Mi - Mnz).power(2) / 50)
        res = inn.sum(axis=1) / (self.sums[i] + self.sums)

        return np.array(res)[:, 0]


class Word2Vec(DiMo):
    """
    Word2Vec wrapper through KeyedVectors class from Gensim package:

    github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/keyedvectors.py
    """

    def __init__(self, name, word2vec_format=False):

        self.model = (
            KeyedVectors.load_word2vec_format(name)
            if word2vec_format else KeyedVectors.load(name).wv
        )

        i2word = {
            i: self.model.index2word[i]
            for i in range(len(self.model.vocab))
        }
        word2i = {word: i for i, word in i2word.items()}

        super(Word2Vec, self).__init__(word2i, i2word)

    def similarity(self, a, b):

        a = (a if type(a) is not int else self.i2word[a])
        b = (b if type(b) is not int else self.i2word[b])

        return self.model.similarity(a, b)

    def similarities(self, word):

        if word is int:
            word = self.i2word[word]

        return self.model.most_similar(word, topn=False)

"""
Wrapper for original Sketch Engine Thesaurus
    ... + some other stuff involving `manatee` or `wmap` package
"""

from collections import defaultdict

import manatee
import wmap


class OutOfVocabError(AttributeError):
    pass


def iter_sentences(corpus_name, struct_name="s", attr_name="word",
                   vocab=None, min_freq=20):
    """
    Yields sentences (as lists of strings) from a corpus compiled in Manatee.

    If you want to restrict the vocabulary in some fancy way,
    provide the set of allowed items through the `vocab` attribute.
    """

    corpus = manatee.Corpus(corpus_name)
    struct = corpus.get_struct(struct_name)
    attr = corpus.get_attr(attr_name)

    if vocab is None:
        vocab = {i: attr.freq(i) for i in range(attr.id_range())}
        vocab = {i: c for i, c in vocab.items() if c >= min_freq}
    else:
        vocab = {i: attr.freq(i) for i in set(vocab)}

    for i in range(struct.size()):
        beg = struct.beg(i)
        end = struct.end(i)

        raw_block = [attr.pos2id(j) for j in range(beg, end)]
        yield [attr.id2str(j) for j in raw_block if j in vocab]


class OriginalThesaurus(object):
    """
    Class that uses the original Sketch Engine Thesaurus
    """

    def __init__(self, corpus_name):

        self.corpus = manatee.Corpus(corpus_name)
        self.wsthes = self.corpus.get_conf("WSTHES")
        self.attr = self.corpus.get_attr(self.corpus.get_conf("WSATTR"))

        self.cached_sims = dict()

    def similarities(self, word):

        if word in self.cached_sims:
            return self.cached_sims[word]

        word_i = self.attr.str2id(word)

        if word_i < 0:
            raise OutOfVocabError("Out-of-vocabulary word '%s'." % word)

        thes = wmap.Thesaurus_f(self.wsthes, word_i)

        sims = dict()

        while not thes.eos():
            sims[self.attr.id2str(thes.getid())] = thes.getscore()
            thes.next()

        self.cached_sims[word] = sims

        return sims

    def most_similar(self, word, topn=10):

        return sorted(self.similarities(word).items(), key=lambda (w, s): s,
                      reverse=True)[:topn]

    def solve_analogy(self, a, b, aa, mul=False):

        sims_a, sims_b, sims_aa = (
            defaultdict(lambda: 0.05, self.similarities(w))
            for w in (a, b, aa)
        )

        results = {
            w: (sims_b[w] * sims_aa[w] / sims_a[w] if mul
                else sims_b[w] + sims_aa[w] - sims_a[w])
            for w in set(sims_b) & set(sims_aa) - {a}
            }.items()

        return (
            None if len(results) == 0
            else max(results, key=lambda (c, s): s)[0]
        )

    def eval_on_dataset(self, dataset, mul=False):

        results = {
            category_label: dict(acc=0.0, oov=0, queries=list())
            for category_label
            in dataset
        }

        for cat, queries in dataset.items():

            for a, b, aa, bbs in queries:

                try:
                    candidate = self.solve_analogy(a, b, aa, mul)
                except OutOfVocabError:
                    results[cat]["oov"] += 1
                    continue

                if candidate in bbs:
                    results[cat]["acc"] += 1

                results[cat]["queries"].append(
                    (a, b, aa, candidate, candidate in bbs)
                )

            # After all queries are processed:
            results[cat]["acc"] /= float(len(queries))

        return results

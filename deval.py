import numpy as np
from copy import deepcopy

# Local imports
from formulas import add as default_formula


class DiMo(object):
    """
    Abstract class for evaluating distributional models on analogy queries

    Subclasses must implement similarity function over words
    """

    def __init__(self, word2i, i2word):

        self.word2i = word2i
        self.i2word = i2word

        # Computing similarities may be expensive
        # We may at least cache them
        self.cached_sims = dict()

    def similarity(self, a, b):
        """
        Returns similarity of two words

        If `a`, `b` are integers, they are treated as word indices
        """
        raise NotImplementedError

    def similarities(self, word):
        """
        Returns vector of similarities of a given word to the whole vocabulary

        If `word` is integer, it is treated as a word index
        """
        raise NotImplementedError

    def eval_analogy(self, dataset, topn=1, exclusion_trick=True,
                     formula=default_formula):
        """
        Evaluates the model on the given dataset.

        `dataset` should be a dictionary
            {category: queries}

        A query is supposed to look like this
            ("paris", "france", "london", {"england", "britain", "uk"})
        """

        results = {
            category_label: dict(
                acc=0.0,  # 0.0--1.0 `topn` accuracy
                acc_top1=0.0,  # 0.0--1.0 top1 accuracy
                oov=0,  # nb of queries containing an oov word
                oovs=set(),  # set of oov words
                queries=list()  # queries with candidate answers
            )
            for category_label in dataset
        }

        self._cache_sims_for_dataset(dataset)

        for cat, queries in dataset.items():

            for a, b, aa, bbs in queries:

                assert type(bbs) == set

                excl_set = {a, aa, b}  # for exclusion trick

                if a not in self.cached_sims:
                    results[cat]["oovs"].add(a)
                if b not in self.cached_sims:
                    results[cat]["oovs"].add(b)
                if aa not in self.cached_sims:
                    results[cat]["oovs"].add(aa)

                try:
                    sims_a = self.cached_sims[a]
                    sims_b = self.cached_sims[b]
                    sims_aa = self.cached_sims[aa]
                except KeyError:
                    results[cat]["oov"] += 1
                    continue

                scores = formula(sims_a, sims_b, sims_aa)
                cand_ids = np.argsort(scores)[::-1][:topn+3]
                cands = [self.i2word[cand_id] for cand_id in cand_ids]

                correct_pos = topn

                #
                # Interpreting Results:
                #
                # correct_pos == 0
                #   ===>  the first candidate is the correct answer
                # correct_pos >= topn
                #   ===>  correct answer not found within topn
                #

                pos = 0
                for cand in cands:

                    if cand in bbs:
                        correct_pos = pos
                        break

                    if exclusion_trick is True and cand in excl_set:
                        continue
                    if exclusion_trick == "aa" and cand == aa:
                        continue

                    pos += 1

                if correct_pos < topn:
                    results[cat]["acc"] += 1

                if correct_pos < 1:
                    results[cat]["acc_top1"] += 1

                results[cat]["queries"].append(
                    (a, b, aa, cands, correct_pos)
                )

            # After all queries are processed:
            results[cat]["acc"] /= float(len(queries))
            results[cat]["acc_top1"] /= float(len(queries))

        return results

    def most_similar(self, positive, negative=None, topn=10, method="add",
                     freq_range=(0, None)):
        if type(positive) in (str, unicode):
            positive = [positive]

        if type(negative) in (str, unicode):
            negative = [negative]
        elif negative is None:
            negative = []

        _from, _to = freq_range[0], freq_range[1]
        if _from is None:
            _from = 0

        sims = {}
        for word in positive + negative:
            assert word in self.word2i
            if word in self.cached_sims:
                sims[word] = self.cached_sims[word]
            else:
                sims[word] = self.similarities(word)

        scores = deepcopy(sims[positive[0]])

        if method == "add":
            for word in positive[1:]:
                scores += sims[word]
            for word in negative:
                scores -= sims[word]
        elif method == "mul":
            for word in positive[1:]:
                scores *= sims[word]
            for word in negative:
                scores /= sims[word] + 0.1
        else:
            raise ValueError("`method` argument must be `add` or `mul`")

        scores = scores[_from:_to]
        indices = np.argsort(scores)[::-1][:topn]

        return [(self.i2word[i + _from], scores[i]) for i in indices]

    def _cache_sims_for_dataset(self, dataset):
        words = set()

        for queries in dataset.values():
            for a, b, aa, __ in queries:
                for word in (a, b, aa):
                    words.add(word)

        for word in words:
            if word in self.word2i and word not in self.cached_sims:
                self.cached_sims[word] = self.similarities(word)


def pairs2queries(pairs, fa=lambda w: w, fb=lambda w: w):
    """
    Converts a list of pairs (a, bs) into analogy queries

    `a` is a word
    `bs` is a word or a tuple of words

    Example pairs:
        ("paris", "france")
        ("london", ("england", "britain", "uk"))

    `fa` may be used to adjust `a`-s
    `fb` may be used to adjust `b`-s

    Example adjustment function:
        "france" -> "France|NOUN"
    """

    new_pairs = []
    queries = []

    for a, bs in pairs:

        if type(bs) in (str, unicode):
            bs = [bs]

        new_pairs.append(
            (fa(a), list(map(fb, bs)))
        )

    for i in range(len(new_pairs)):
        for j in range(len(new_pairs)):
            if i != j:
                a, bs = new_pairs[i]
                aa, bbs = new_pairs[j]
                queries.append((a, bs[0], aa, set(bbs)))

    return queries

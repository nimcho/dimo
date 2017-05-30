#!/usr/bin/python
"""
Word Sketch Data ---> Word-(Relation,Word) Matrix
=================================================

Script for transforming word sketch data into a sparse target-context
co-occurrence matrix with the same parameters as original wm2thes.

To save memory, the script does two-passes over the word sketch data.
"""

import sys
import pickle
import numpy as np

# Sketch Engine imports
import wmap
import manatee

# ------------------------------------------------------------------------------

# Only (word, rel, col) triples exceeding these values shall pass:

TARGET_COUNT = 20
TRIPLE_COUNT = 1
TRIPLE_SCORE = 0

# ------------------------------------------------------------------------------


def iter_word_sketches(corpus_id):
    """
    Yields (word, rel_id, coll_id, score) quadruples.
    """

    corpus = manatee.Corpus(corpus_id)
    attr = corpus.get_attr(corpus.get_conf('WSATTR'))
    wsbase = corpus.get_conf('WSBASE')

    wmap1 = wmap.WMap(wsbase, 0, 0, 0, corpus_id)

    while True:  # over targets
        if wmap1.getcnt() > TARGET_COUNT:
            word = attr.id2str(wmap1.getid())
            wmap2 = wmap1.nextlevel()

            while True:  # over target's relations
                rel_id = wmap2.getid()
                wmap3 = wmap2.nextlevel()

                while True:  # over target's relation's collocates
                    coll_id = wmap3.getid()
                    count = wmap3.getcnt()
                    rank = wmap3.getrnk()

                    if count > TRIPLE_COUNT and rank > TRIPLE_SCORE:
                        yield (word, rel_id, coll_id, rank)

                    if not wmap3.next():
                        break

                if not wmap2.next():
                    break

        if not wmap1.next():
            break


def main():

    if len(sys.argv) != 3:
        sys.stderr.write("Usage: python wm2thes.py CORPUS_NAME OUTPUT_NAME\n")
        sys.exit(1)

    corpus_name = sys.argv[1]
    output_name = sys.argv[2]

    #

    nb_cells = 0
    targets = set()
    contexts = set()
    for word, rel_id, col_id, ___ in iter_word_sketches(corpus_name):
        nb_cells += 1
        targets.add(word)
        contexts.add((rel_id, col_id))

    # TODO: targets sorted by freqs

    target2i = {target: i for i, target in enumerate(targets)}
    context2i = {(rel, coll): i for i, (rel, coll) in enumerate(contexts)}

    # Sparse target-context co-occurrence matrix
    rows = np.zeros((nb_cells,), dtype=np.int64)
    cols = np.zeros((nb_cells,), dtype=np.int64)
    vals = np.zeros((nb_cells,), dtype=np.float32)

    for i, (word, rel_id, coll_id, score) in enumerate(iter_word_sketches(corpus_name)):
        rows[i] = target2i[word]
        cols[i] = context2i[(rel_id, coll_id)]
        vals[i] = score

    np.save(output_name + "-rows.npy", rows)
    np.save(output_name + "-cols.npy", cols)
    np.save(output_name + "-vals.npy", vals)

    with open(output_name + "-target2i.pickle", "w") as f:
        pickle.dump(target2i, f)


if __name__ == "__main__":
    main()

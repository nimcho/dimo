# DiMo.  Distributional Models Evaluation

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/nimcho/dimo/blob/master/LICENSE)

DiMo is a collection of scripts for my bachelor's thesis *Comparison and Evaluation of Models for Distributional Semantics*.

Take a look at notebooks on the thesis's official website to see how these scripts can be used:

 - [nlp.fi.muni.cz/projekty/dimo](https://nlp.fi.muni.cz/projekty/dimo/)

**Notice!**  Parts of the code require internal [Sketch Engine](https://www.sketchengine.co.uk/)'s packages `manatee` and `wmap`.

Other required packages are:

 - numpy
 - scipy
 - gensim
 - sklearn

The code runs on python 2.7.

## Models

### Sketch Engine Thesaurus (SkEThes)

Unlike the original implementation, the one in this project operates directly on a co-occurrence matrix.

If you have a corpus with compiled word sketches (let's say it is called *bnc2*), use `wm2thes.py` script to create such a matrix:

```bash
python wm2thes.py bnc2 bnc2-matrix
```

This creates 4 files representing a sparse *word x (relation, word)* matrix:

    - bnc2-matrix-word2i.pickle  # dictionary: words to indices
    - bnc2-matrix-rows.npy       # row indices
    - bnc2-matrix-cols.npy       # col indices
    - bnc2-matrix-vals.npy       # values

Now that you have the matrix, you may decide which similarity measure to use.

```python
from models import SkEThesSKE, SkEThesCOS

model_ske = SkEThesSKE("bnc2-matrix")
model_cos = SkEThesCOS("bnc2-matrix")
```

Now you can call functions like `similarity`, `similarities`, `most_similar` or `eval_analogy` to evaluate the models on datasets of analogy queries.

### Word-Word Co-Occurrence Matrix

If you have a corpus in text file (one line -- one sentence), you may create a similar model with linear contexts (weighted symmetric context window):

```bash
python coocs.py plain-bnc.txt plain-bnc-matrix 20 5
```

 - 20 is the minimum word frequency 
 - 5 is the context window size

The matrix will contain raw co-occurrence counts, so you may consider using some weighting.

```python
from models import SkEThesCOS
from weightings import ppmi

model_ske = SkEThesSKE("plain-bnc-matrix", weighting=ppmi)
```

### Word2Vec

For Word2Vec models, this project wraps over gensim package.  Everything that you can open with:

```python
from gensim.models import Word2Vec

model = Word2Vec(model_name)
```

... you can open also with:

```python
from models import Word2Vec

model = Word2Vec(model_name)
```

The interface as well as the evaluation script stays the same as in `SkEThesXXX`.

## Evaluation

```python
evaluation = model.eval_analogy(dataset)
```

Dataset is a dictionary `category: list_of_queries`.  Each query should be a tuple like:

```python
("paris", "france", "london", {"england", "britain", "uk"})
```

You may configure the evaluation in various ways:

```python
from formulas import mul

my_mul = lambda a, b, aa: mul(a, b, aa, coeff=0.05)
evaluation = model.eval_analogy(dataset, topn=5, exclusion_trick=False, formula=my_mul)
```

And see the results:

```python
evaluation["acc"]  # 0.0--1.0
evaluation["acc_top1"]  # 0.0--1.0
evaluation["oov"]  # nb of queries containing an oov word
evaluation["oovs"]  # set of oov words
queries=list()  # queries with candidate answers
```

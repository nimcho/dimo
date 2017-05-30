"""
Formulas for Resolving Analogy Queries

See the section 4.2 for more information on this.
"""

# Each input `a`, `b`, `aa` should be a vector of similarities
# of corresponding word to the whole vocab


def add(a, b, aa):
    return b - a + aa


def mul(a, b, aa, epsilon=0.1):
    return b * aa / (a + epsilon)


def only_aa(a, b, aa):
    return aa

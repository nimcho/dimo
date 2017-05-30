"""
Evaluation datasets for word analogy queries
"""

from glob import glob

#
# capital-common-countries dataset
# excluding (london, england) pair
#
ccc_pairs = [
    ("athens", "greece"),
    ("baghdad", "iraq"),
    ("bangkok", "thailand"),
    ("beijing", "china"),
    ("berlin", "germany"),
    ("bern", "switzerland"),
    ("cairo", "egypt"),
    ("canberra", "australia"),
    ("hanoi", "vietnam"),
    ("havana", "cuba"),
    ("helsinki", "finland"),
    ("islamabad", "pakistan"),
    ("kabul", "afghanistan"),
    ("madrid", "spain"),
    ("moscow", "russia"),
    ("oslo", "norway"),
    ("ottawa", "canada"),
    ("paris", "france"),
    ("rome", "italy"),
    ("stockholm", "sweden"),
    ("tehran", "iran"),
    ("tokyo", "japan"),
]


#
# BATS Dataset
# ------------------------------------------------------------------------------
#
# `bats_path` on Alba server:
# /nlp/projekty/dimo/datasets/bats/BATS_3.0
#
# It contains plain lowercased words.  If you need lemposes instead,
# use corresponding adjustment functions from the `conv.py` script.
#
# Note: L07, L08 and L10 mix up POS tags, those 3 are hard-annotated
# in /nlp/projekty/dimo/datasets/bats/BATS_3.0_pos
# :( still use adj. funcs. for others, they are still plain there
#
def get_bats(bats_path):

    bats = {}  # dictionary {category_name: list_of_pairs}

    for bats_file in glob(bats_path + "/*/*"):

        category_name = bats_file.split("/")[-1].split(" ")[0].lower()
        bats[category_name] = []

        with open(bats_file) as f:
            for line in f:
                parts = line.strip("\r\n /").split("\t")
                bats[category_name].append(
                    (parts[0],
                     parts[1].split("/"))
                )

    return bats

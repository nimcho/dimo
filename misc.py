import pickle
from gzip import open as gzip_open
from collections import defaultdict

# ------------------------------------------------------------------------------
# Openers
#

file_openers = {
    "vert": open,
    "txt": open,
    "gz": gzip_open,
}


def get_opener(file_path):
    """ Gets a proper opening function for certain file types. """
    return file_openers[file_path.split(".")[-1]]


def use_opener(file_path, mode="r"):
    return get_opener(file_path)(file_path, mode)


# ------------------------------------------------------------------------------


class LineCorpus(object):

    def __init__(self, file_name):

        self.file_name = file_name

    def __iter__(self):

        with use_opener(self.file_name) as f:
            for line in f:
                sentence = line.strip("\n\r\t ").split()
                if len(sentence) > 0:
                    yield sentence


def corpus2vocab(corpus):

    vocab = defaultdict(lambda: 0)

    for sentence in corpus:
        for word in sentence:
            vocab[word] += 1

    return vocab


def dump_sentences(sentences, output_file):
    with use_opener(output_file, "w") as f:
        for sentence in sentences:
            f.write(" ".join(sentence) + "\n")

# ------------------------------------------------------------------------------


def save_report(report, dataset_name, model_name, formula, directory="reports/"):
    parts = [dataset_name, model_name, formula]
    name = ".".join(parts)
    with open(directory + "/" + name + ".pickle", "w") as f:
        pickle.dump(file=f, obj=report)


def load_report(name):
    with open(name) as f:
        return pickle.load(file=f)


from gensim.models import Word2Vec

from tokenlearn.logging_config import setup_logging


class Sentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            yield line.lower().split()


if __name__ == "__main__":
    setup_logging()
    sent = Sentences("corpus_10m.txt")
    model = Word2Vec(sent, min_count=2, vector_size=300, workers=3, sg=1, max_vocab_size=400_000)

    model.wv.save_word2vec_format("corpus_10m.vec")

import nltk
from nltk.corpus import brown
from cPickle import dump, load
import os


def build_tagger():
    brown_tagged_sents = brown.tagged_sents()
    t0 = nltk.DefaultTagger('NN')
    t1 = nltk.UnigramTagger(brown_tagged_sents, backoff=t0)
    t2 = nltk.BigramTagger(brown_tagged_sents, backoff=t1)
    with open('tagger.pickle', 'wb') as f:
        dump(t2, f, -1)
    return t2


def tagger():
    if os.path.exists('tagger.pickle'):
        with open('tagger.pickle', 'rb') as f:
            tagger = load(f)
    else:
        tagger = build_tagger()
    return tagger


if __name__ == "__main__":
    tagger = tagger()

    print tagger.evaluate(brown.tagged_sents(categories='news'))
    print tagger.evaluate(brown.tagged_sents(categories='reviews'))

import os
import nltk
from collections import defaultdict
from noun_phrases_extraction import extract_noun_phrases
import filtering


def sentenceSumSentiment(taggedSentence):
    """
    Returns the sum sentiment of a TaggedSentence's features
    """
    return sum([feature.sign for feature in taggedSentence.features])


def sentenceTernarySentiment(taggedSentence):
    """
    Returns -1, 0, 1 for negative, neutral, and positive sentiment, respectivly
    """
    sumFeatures = sentenceSumSentiment(taggedSentence)
    if sumFeatures < 0:
        return -1
    elif sumFeatures > 0:
        return 1
    else:
        return 0


def normalize(d, target=1.0):
    factor = target / (sum(d.itervalues()) + .000001)
    for key, val in d.iteritems():
        d[key] = val * factor
    return d


def loadWordSentimentDict(dictPath='word_sentiment.dict'):
    """
    Loads word sentiment dict from word_sentiment.dict file
    """
    sentimentDict = defaultdict(lambda: 0)
    with open(os.path.abspath(dictPath), 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                pass
            else:
                word, sentiment = line.split('\t')
                sentimentDict[word] = int(sentiment)
    return normalize(sentimentDict)


def buildWordSentimentDict(taggedReviews,
                           applyFn=sentenceSumSentiment,
                           filterFn=filtering.chainFilter(filtering.lower,
                                                          filtering.removeStopwords)):
    """
    Builds a dictionary of word sentiments from training data by taking the
    running average of applying fn (defaults to sentenceSumSentiment). Filters
    out words contained in the filterDict argument.
    """
    sentimentDict = defaultdict(lambda: 0)
    for taggedReview in taggedReviews:
        for taggedSentence in taggedReview:
            tokenizedSentence = filtering.tokenize(taggedSentence.sentence)
            filteredSentence = filterFn(tokenizedSentence)
            for word in filteredSentence:
                sentimentDict[word] += applyFn(taggedSentence)
    return normalize(sentimentDict)


def buildNounPhraseDict(taggedReviews,
                        applyFn=sentenceSumSentiment):
    nounPhraseDict = defaultdict(lambda: 0)
    for taggedReview in taggedReviews:
        for taggedSentence in taggedReview:
            for np in extract_noun_phrases(taggedSentence.sentence):
                nounPhraseDict[np] += applyFn(taggedSentence)
    return normalize(nounPhraseDict)


def buildNGramDict(taggedReviews,
                   n=1,
                   applyFn=sentenceSumSentiment,
                   filterFn=filtering.chainFilter(filtering.lower,
                                                  filtering.removeStopwords)):
    ngramDict = defaultdict(lambda: 0)
    for taggedReview in taggedReviews:
        for taggedSentence in taggedReview:
            sentenceSentiment = applyFn(taggedSentence)
            for ngram in nltk.ngrams(filterFn(filtering.tokenize(taggedSentence.sentence)), n):
                ngramDict[ngram] += ngramDict[ngram]
    return normalize(ngramDict)


def main():
    pass

if __name__ == "__main__":
    main()

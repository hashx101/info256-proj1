import os
import nltk
from nltk import ngrams
from nltk.tokenize import wordpunct_tokenize
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
    return sentimentDict


def buildWordSentimentDict(taggedReviews,
                           applyFn=sentenceSumSentiment,
                           filterFn=filtering.chainFilter(filtering.lower,
                                                          filtering.remove_stopwords)):
    """
    Builds a dictionary of word sentiments from training data by taking the
    running average of applying fn (defaults to sentenceSumSentiment). Filters
    out words contained in the filterDict argument.
    """
    nounPhraseDict = defaultdict(lambda: 0)
    for taggedReview in taggedReviews:
        for taggedSentence in taggedReview:
            tokenizedSentence = wordpunct_tokenize(taggedSentence.sentence)
            filteredSentence = filterFn(tokenizedSentence)
            for word in filteredSentence:
                nounPhraseDict[word] = (nounPhraseDict[word] + applyFn(taggedSentence)) / 2
    return nounPhraseDict


def buildNounPhraseDict(taggedReviews,
                        applyFn=sentenceSumSentiment):
    nounPhraseDict = defaultdict(lambda: 0)
    for taggedReview in taggedReviews:
        for taggedSentence in taggedReview:
            for np in extract_noun_phrases(taggedSentence.sentence):
                nounPhraseDict[np] = (nounPhraseDict[np] + applyFn(taggedSentence)) / 2
    return nounPhraseDict


def buildNGramDict(taggedReviews, n=1):
    ngramDict = defaultdict(lambda: 0)
    for taggedReview in taggedReviews:
        for taggedSentence in taggedReview:
            for ngram in ngrams(wordpunct_tokenize(taggedSentence.sentence), n):
                ngramDict[ngram] = sum([feature.sign for feature in taggedSentence.features])
    return ngramDict


def main():
    pass

if __name__ == "__main__":
    main()
    
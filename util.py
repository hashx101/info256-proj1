import os
from nltk import ngrams
from nltk.tokenize import wordpunct_tokenize
from collections import defaultdict
from noun_phrases_extraction import extract_noun_phrases


def getWordSentimentDict():
    sentimentDict = defaultdict(lambda: 0)
    with open('word_sentiment.dict', 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                pass
            else:
                word, sentiment = line.split('\t')
                sentimentDict[word] = int(sentiment)
    return sentimentDict


sentimentDict = getWordSentimentDict()
def sumSentiment(sequence):
    return reduce(lambda total, word: total + sentimentDict[word], sequence, 0)


def buildNGramDict(taggedReviews, n=1):
    ngramDict = defaultdict(lambda: 0)
    for taggedReview in taggedReviews:
        for taggedSentence in taggedReview:
            for ngram in ngrams(wordpunct_tokenize(taggedSentence.sentence), n):
                ngramDict[ngram] = sum([feature.sign for feature in taggedSentence.features])
    return ngramDict

def buildNounPhraseDict(taggedReviews):
    nounPhraseDict = defaultdict(lambda: 0)
    for taggedReview in taggedReviews:
        for taggedSentence in taggedReview:
            for np in extract_noun_phrases(taggedSentence.sentence):
                nounPhraseDict[np] = 1 if 0 < sum([feature.sign for feature in taggedSentence.features]) else -1
    return nounPhraseDict


def main():
    print(getWordSentimentDict())


if __name__ == "__main__":
    main()
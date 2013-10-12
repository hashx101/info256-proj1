from __future__ import division
import parse
import util

import nltk
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier

import collections
import re
import os
import glob
from pprint import pprint


###############################################################################
## Feature generation
###############################################################################

definedFns = []
sentiments = util.getWordSentimentDict()

def total_sentiment(inp):
    total = 0
    for word in map(lambda w: w.lower().strip(), inp.split(' ')):
        if word in sentiments:
            total += sentiments[word]
    return total

def num_positive_sentiment_words(inp):
    totalPos = 0
    for word in map(lambda w: w.lower().strip(), inp.split(' ')):
        if word in sentiments and sentiments[word] > 0:
            totalPos += 1
    return totalPos

def num_negative_sentiment_words(inp):
    totalNeg = 0
    for word in map(lambda w: w.lower().strip(), inp.split(' ')):
        if word in sentiments and sentiments[word] < 0:
            totalNeg += 1
    return totalNeg

for fn in [('total_sentiment', total_sentiment),
           ('num_positive_sentiment_words', num_positive_sentiment_words),
           ('num_negative_sentiment_words', num_negative_sentiment_words)]:
    definedFns.append(fn)

###############################################################################
## Classifier
###############################################################################

def taggedReviews():
    """Generates a list of tagged sentence/sentiment tuples for training
    our classifier"""
    directory = "data/training"
    os.chdir(directory)
    files = glob.glob("*.txt")

    taggedReviews = []
    for filename in files:
        parsedReviews = parse.parse_file(os.path.join(os.curdir, filename))
        reviewText = ""
        for review in parsedReviews:
            for taggedSentence in review:
                featureSentiment = 0
                for feature in taggedSentence.features:
                    featureSentiment += feature.sign * feature.magnitude
                taggedReviews.append((taggedSentence.sentence, featureSentiment))
    return taggedReviews


def applyFeatures(inp, *vectorFns):
    featureDict = {}
    for fn in vectorFns:
        out = fn(inp)
        featureDict[fn.__name__] = out
    return featureDict


def buildClassifier(inp,
                    holdoutRatio=0,
                    featureList=map(lambda tup: tup[1], definedFns)):
    processedFeatures = [(applyFeatures(name, *featureList), tag) for name, tag in inp]
    trainSet = processedFeatures[:int(len(processedFeatures) * (1 - holdoutRatio))]
    holdoutSet = processedFeatures[int(len(processedFeatures) * holdoutRatio):]
    classifier = SklearnClassifier(LinearSVC()).train(trainSet)
    return classifier, nltk.classify.accuracy(classifier, holdoutSet)

def main():
    pprint(map(lambda x: x[0], definedFns))
    print("{} features".format(len(definedFns)))
    c, r = buildClassifier(taggedReviews(), 0)
    print r
    return c, r

if __name__ == '__main__':
    main()
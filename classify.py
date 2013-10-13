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

originalDir = os.curdir
###############################################################################
## Feature generation
###############################################################################

os.chdir('data/training')
files = glob.glob("*.txt")
taggedReviews = []
for filename in files:
    taggedReviews += parse.parse_file(filename)
os.chdir('../..')

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

for n, prefix in zip(range(1,6),['uni', 'bi', 'tri', 'quadra', 'penta']):
    exec("{}gramDict = util.buildNGramDict(taggedReviews, {})".format(prefix, n))
    exec("""def {}gram_score(inp):\n\ttotal = 0\n\tfor {}gram in nltk.ngrams(inp, 
        {}):\n\t\ttotal += {}gramDict[{}gram]\n\treturn total""".format(prefix,
                                                                         prefix,
                                                                         n,
                                                                         prefix,
                                                                         prefix))
    fnName = "{}gram_score".format(prefix)
    definedFns.append((fnName, eval(fnName)))

for fn in [('total_sentiment', total_sentiment),
           ('num_positive_sentiment_words', num_positive_sentiment_words),
           ('num_negative_sentiment_words', num_negative_sentiment_words)]:
    definedFns.append(fn)

###############################################################################
## Classifier
###############################################################################

def taggedReviews(directory="data/training"):
    """Generates a list of tagged sentence/sentiment tuples for training
    our classifier"""
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
    processedFeatures = [(applyFeatures(text, *featureList), tag) for text, tag in inp]
    trainSet = processedFeatures[:int(len(processedFeatures) * (1 - holdoutRatio))]
    holdoutSet = processedFeatures[int(len(processedFeatures) * holdoutRatio):]
    classifier = SklearnClassifier(LinearSVC()).train(trainSet)
    if len(holdoutSet) > 0:
        nltk.classify.accuracy(classifier, holdoutSet)
    return classifier


def main():
    pprint(map(lambda x: x[0], definedFns))
    print("{} features".format(len(definedFns)))
    c = buildClassifier(taggedReviews(), 0)
    holdoutSet = taggedReviews('/home/alexm/info256/proj1/data/heldout')
    print nltk.classify.accuracy(c, [(applyFeatures(text, *map(lambda tup: tup[1], definedFns)), tag) for text, tag in holdoutSet])
    return c

if __name__ == '__main__':
    main()
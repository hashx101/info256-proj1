from __future__ import division
import parse

import nltk
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier

import collections
import re
import os
import glob


###############################################################################
## Feature generation
###############################################################################

definedFns = []

def num_vowels(inp):
    return len([c for c in inp if c.lower() in 'aeiouy'])

for fn in [('num_vowels', num_vowels)]:
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
                for feature in taggedSentence.features:
                    taggedReviews.append((taggedSentence.sentence, feature.sign))
    return taggedReviews


def applyFeatures(inp, *vectorFns):
    featureDict = {}
    for fn in vectorFns:
        out = fn(inp)
        featureDict[fn.__name__] = out
    return featureDict


def buildClassifier(inp,
                    holdoutRatio=.1,
                    featureList=map(lambda tup: tup[1], definedFns)):
    processedFeatures = [(applyFeatures(name, *featureList), tag) for name, tag in inp]
    trainSet = processedFeatures[:int(len(processedFeatures) * (1 - holdoutRatio))]
    holdoutSet = processedFeatures[int(len(processedFeatures) * holdoutRatio):]
    classifier = SklearnClassifier(LinearSVC()).train(trainSet)
    return classifier, nltk.classify.accuracy(classifier, holdoutSet)

def main():
    # pprint("{} features".format(len(definedFns)))
    # c, r = buildClassifier(taggedNames(), 0)
    # print r
    # return c, r
    print taggedReviews()

if __name__ == '__main__':
    main()
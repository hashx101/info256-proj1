from __future__ import division
import parse
import util
import filtering

import nltk
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from nltk.classify import SklearnClassifier

import collections
import re
import os
import glob
from pprint import pprint

###############################################################################
## Feature generation
###############################################################################
originalDir = os.path.abspath(os.curdir)
definedFns = []
loadedSentimentDicts = []
taggedSentenceEvaluationFunctions = [('sum', 'util.sentenceSumSentiment'),
                                     ('ternary', 'util.sentenceTernarySentiment')]

###### Load tagged reviews ####################################################
os.chdir('data/training')
files = glob.glob("*.txt")
taggedReviews = []
for filename in files:
    taggedReviews += parse.parse_file(filename)
os.chdir(originalDir)

###### Compile/load sentiment dictionaries ####################################
staticSentimentDicts = [('afinn96', 'dicts/afinn-96.dict'),
                        ('afinn111', 'dicts/afinn-111.dict'),
                        ('nielsen2009', 'dicts/Nielsen2009Responsible_emotion.dict'),
                        ('nielsen2009', 'dicts/Nielsen2010Responsible_english.dict')]

# load static dicts
for name, path in staticSentimentDicts:
    dictSym = "{}SentimentsDict".format(name)
    exec("{} = util.loadWordSentimentDict(os.path.join(originalDir, '{}'))".format(dictSym,
                                                                                   path))
    loadedSentimentDicts.append((dictSym, eval(dictSym)))

# compile learned dicts using various apply fucntions

for name, fn in taggedSentenceEvaluationFunctions:
    exec("learned_{}_sentiments = util.buildWordSentimentDict(taggedReviews, applyFn={})".format(name, fn))
    dictName = "learned_{}_sentiments".format(name)
    loadedSentimentDicts.append((dictName, eval(dictName)))

filterFn = filtering.chainFilter(filtering.lower,
                                 filtering.lemmatize,
                                 filtering.removeStopwords)
for name, sentimentDict in loadedSentimentDicts:
    exec("""def total_sentiment_{0}(inp):
        total = 0
        for word in filterFn(filtering.tokenize(inp)):
            if word in {0}:
                total += {0}[word]
        return total""".format(name))
    fnName = "total_sentiment_{}".format(name)
    definedFns.append((fnName, eval(fnName)))

    exec("""def num_positive_sentiment_words_{0}(inp):
            totalPos = 0
            for word in filterFn(filtering.tokenize(inp)):
                if word in {0} and {0}[word] > 0:
                    totalPos += 1
            return totalPos
        """.format(name))
    fnName = "num_positive_sentiment_words_{}".format(name)
    definedFns.append((fnName, eval(fnName)))

    exec("""def num_negative_sentiment_words_{0}(inp):
    totalNeg = 0
    for word in filterFn(filtering.tokenize(inp)):
        if word in {0} and {0}[word] < 0:
            totalNeg += 1
    return totalNeg""".format(name))
    fnName = "num_negative_sentiment_words_{}".format(name)
    definedFns.append((fnName, eval(fnName)))

for n, prefix in zip(range(1,6),['uni', 'bi', 'tri', 'quadra', 'penta']):
    exec("{}gramDict = util.buildNGramDict(taggedReviews, {})".format(prefix, n))
    exec("""def {0}gram_score(inp):\n\ttotal = 0\n\tfor {0}gram in nltk.ngrams(inp, 
        {1}):\n\t\ttotal += {0}gramDict[{0}gram]\n\treturn int(total)""".format(prefix,
                                                                           n))
    fnName = "{}gram_score".format(prefix)
    definedFns.append((fnName, eval(fnName)))


for name, fnName in taggedSentenceEvaluationFunctions:
    dictName = "nounphrase_{}_dict".format(name)
    exec("{} = util.buildNounPhraseDict(taggedReviews, applyFn={})".format(dictName, fnName))
    exec("""def nounphrase_{}_score(inp):
            return {}[inp]
         """.format(name, dictName))
    featureName = "nounphrase_{}_score".format(name)
    definedFns.append((featureName, eval(featureName)))


###############################################################################
## Classifier
###############################################################################

def taggedReviews(directory="data/training"):
    """Generates a list of tagged sentence/sentiment tuples for training
    our classifier. Only takes lines where there are either 0 features, or all
    positive or negative features."""
    startingDir = os.curdir
    os.chdir(os.path.abspath(directory))
    files = glob.glob("*.txt")

    taggedReviews = []
    for filename in files:
        parsedReviews = parse.parse_file(os.path.abspath(os.path.join(os.curdir, filename)))
        reviewText = ""
        for review in parsedReviews:
            for taggedSentence in review:
                if len(taggedSentence.features) == 0:
                    taggedReviews.append((taggedSentence.sentence, 0))
                else:
                    firstSentiment = taggedSentence.features[0].sign
                    allSame = True
                    for feature in taggedSentence.features:
                        if feature.sign != firstSentiment:
                            allSame = False
                    if allSame:
                        taggedReviews.append((taggedSentence.sentence, firstSentiment))
    os.chdir(os.path.abspath(startingDir))
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
    classifier = SklearnClassifier(LinearSVC(dual=False)).train(trainSet)
    if len(holdoutSet) > 0:
        nltk.classify.accuracy(classifier, holdoutSet)
    print("Trained accuracy: {}".format(nltk.classify.accuracy(classifier, trainSet)))
    return classifier


def main():
    pprint(map(lambda x: x[0], definedFns))
    print("{} features".format(len(definedFns)))
    c = buildClassifier(taggedReviews(), 0)
    holdoutSet = taggedReviews('/home/alexm/info256/proj1/data/heldout')
    print "Holdout accuracy: {}".format(nltk.classify.accuracy(c,
                                                               [(applyFeatures(text,
                                                                               *map(lambda tup: tup[1],
                                                                                    definedFns)),
                                                                               tag) for text, tag in holdoutSet]))
    return c

if __name__ == '__main__':
    main()

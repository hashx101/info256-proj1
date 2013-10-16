from __future__ import division
import parse
import util
import filtering

import nltk
from nltk.classify import MaxentClassifier

import collections
import re
import os
import glob
from pprint import pprint
import sys

###############################################################################
## Feature generation
###############################################################################
originalDir = os.path.abspath(os.curdir)
definedFns = []


def tagReviews(directory="data/training", output=False):
    """Generates a list of tagged sentence/sentiment tuples for training
    our classifier. Only takes lines where there are either 0 features, or all
    positive or negative features."""
    startingDir = os.getcwd()
    os.chdir(os.path.abspath(directory))
    files = glob.glob("*.txt")
    if output:
        outputFile = open('../../output.txt', 'w')

    taggedReviews = []
    for filename in files:
        lineNumber = 0
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
                        if output:
                            outputFile.write("%s\t%d\t%d\n" % (filename, lineNumber, firstSentiment))
                lineNumber += 1
    if output:
        outputFile.close()
        print "Output stored for %s" % directory

    os.chdir(os.path.abspath(startingDir))
    return taggedReviews


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
    
    # naive bayes
    classifier = nltk.NaiveBayesClassifier.train(trainSet)
    
    # max ent
    #algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    #classifer = MaxentClassifier.train(trainSet, algorithm)
    
    if len(holdoutSet) > 0:
        nltk.classify.accuracy(classifier, holdoutSet)
    print("Trained accuracy: {}".format(nltk.classify.accuracy(classifier, trainSet)))
    return classifier


def grade(directory, classifier):
    os.chdir(directory)
    files = glob.glob("*.txt")
    output = ""
    for filename in files:
        with open(filename) as f:
            for i, line in enumerate(f.readlines()):
                classification = classifier.classify(applyFeatures(line, *map(lambda tup: tup[1], definedFns)))
                if "[t]" in line:
                    classification = 0
                else:
                    line = line[2:]
                outputLine = "{} {} {}\n".format(filename, i + 1, classification)
                print outputLine
                output += outputLine
    with open(os.path.join(directory, 'g_10_output.txt'), 'w') as f:
        f.write(output)

def main(directory):
    c = buildClassifier(tagReviews(), 0)
    grade(directory, c)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Error: Incorrect number of arguments."
    main(sys.argv[1])

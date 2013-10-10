from __future__ import division
import nltk
from nltk.corpus import names as namelist
import random
from pprint import pprint
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from hyphen import Hyphenator
import hyphen.dictools
import collections
import re


###############################################################################
## Feature generation
###############################################################################

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
VOWELS = 'aeiouy'
CONSONANTS = 'bcdfghjklmnpqrstvwxyz'
definedFns = []

def extractSylables(words, syllableConstraints=[]):
    nameDict = {}
    [nameDict.update({word: 1}) for word in map(lambda w: w.lower(), nltk.corpus.names.words())]
    
    syllableConstraints.append(lambda s: len(s) > 1)
    syllableConstraints.append(lambda s: len(s) < 5)
    syllableConstraints.append(lambda s: s not in nameDict)

    syllableCounts, hyphenator = collections.defaultdict(lambda: 1), Hyphenator('en_US')
    for word in filter(lambda w: w.lower() not in nameDict, words):
        syllables = hyphenator.syllables(unicode(word.lower()))
        for fn in syllableConstraints:
            syllables = filter(fn, syllables)
        for syllable in syllables:
            syllableCounts[syllable] += 1
    topSyllables = sorted(syllableCounts.items(), key=lambda t: -t[1])
    return map(lambda t: t[0], topSyllables[:5000])


for syllable in extractSylables(nltk.corpus.brown.words()):
    if re.match("^[A-Za-z][a-zA-Z]*$", syllable):
        try:
            exec("""def syl_{}(inp):\treturn "{}" in inp""".format(syllable, syllable))
            fnName = "syl_{}".format(syllable)
            definedFns.append((fnName, eval(fnName)))
        except:
            print('Skipping syllable {}'.format(syllable))

for letter in ALPHABET:
    exec("""def startswith_{}(inp):\treturn inp.startswith("{}")""".format(letter, letter))
    fnName = "startswith_{}".format(letter)
    definedFns.append((fnName, eval(fnName)))
    exec("""def endswith_{}(inp):\treturn inp.endswith("{}")""".format(letter, letter))
    fnName = "endswith_{}".format(letter)
    definedFns.append((fnName, eval(fnName)))

def num_vowels(inp):
    return len([c for c in inp if c.lower() in VOWELS])

def vowels_ratio(inp):
    return num_vowels(inp) / len(inp)

def num_consonants(inp):
    return len([c for c in inp if c.lower() in CONSONANTS])

def consonants_ratio(inp):
    return num_consonants(inp) / len(inp)

for fn in [('num_vowels', num_vowels),
           ('vowels_ratio', vowels_ratio),
           ('num_consonants', num_consonants),
           ('consonants_ratio', consonants_ratio)]:
    definedFns.append(fn)

###############################################################################
## Classifier
###############################################################################

def taggedNames():
    names = ([(name.lower(), 'male') for name in namelist.words('male.txt')] +
             [(name.lower(), 'female') for name in namelist.words('female.txt')])
    random.shuffle(names)
    return names


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
    return classifier, nltk.classify.accuracy(classifier, processedFeatures)

def maxScore():
    male = set([name.lower() for name in namelist.words('male.txt')])
    female = set([name.lower() for name in namelist.words('female.txt')])
    return 1 - (len(male.intersection(female)) / (len(male) + len(female)))

def main():
    pprint("{} features".format(len(definedFns)))
    c, r = buildClassifier(taggedNames(), 0)
    print r
    return c, r

if __name__ == '__main__':
    main()
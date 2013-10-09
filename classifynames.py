from __future__ import division
import nltk
from nltk.corpus import names as namelist
import random
from pprint import pprint
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from primesieve import genPrimes


ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
VOWELS = 'aeiouy'
CONSONANTS = 'bcdfghjklmnpqrstvwxyz'

definedFns = []

primeGen = genPrimes()
convDict = dict()
[convDict.update({letter: primeGen.next()}) for letter in ALPHABET]

for consonant1 in CONSONANTS:
        for vowel1 in VOWELS:
            exec("""def syl_{}{}(inp):\treturn "{}{}" in inp""".format(consonant1, vowel1, consonant1, vowel1))
            fnName = "syl_{}{}".format(consonant1, vowel1)
            definedFns.append((fnName, eval(fnName)))
            convDict.update({consonant1 + vowel1: primeGen.next()})
            for vowel2 in VOWELS:
                exec("""def syl_{}{}{}(inp):\treturn "{}{}{}" in inp""".format(consonant1, vowel1, vowel2, consonant1, vowel1, vowel2))
                fnName = "syl_{}{}{}".format(consonant1, vowel1, vowel2)
                definedFns.append((fnName, eval(fnName)))
                exec("""def syl_{}{}(inp):\treturn "{}{}" in inp""".format(vowel1, vowel2, vowel1, vowel2))
                fnName = "syl_{}{}".format(vowel1, vowel2)
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

def taggedNames():
    names = ([(name.lower(), 'male') for name in namelist.words('male.txt')] +
             [(name.lower(), 'female') for name in namelist.words('female.txt')])
    random.shuffle(names)
    return names


def applyFeatures(inp, *vectorFns):
    featureDict = {}
    for fn in vectorFns:
        out = fn(inp)
        if type(out) == str:
            if out in convDict:
                out = convDict[fn(inp)]
            else:
                out = 0
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

def max_score():
    male = set([name.lower() for name in namelist.words('male.txt')])
    female = set([name.lower() for name in namelist.words('female.txt')])
    return 1 - (len(male.intersection(female)) / (len(male) + len(female)))


def main():
    pprint(map(lambda x: x[0], definedFns))
    pprint("{} features".format(len(definedFns)))
    c, r = buildClassifier(taggedNames(), 0)
    print r
    return c, r



if __name__ == '__main__':
    main()
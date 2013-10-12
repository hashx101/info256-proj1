from collections import namedtuple
import re
import pprint
import glob
import os


class Feature(object):

    def __init__(self, feature='', sign='+', magnitude=0, tags=[]):
        self.feature = feature
        self.sign = int(sign + '1')
        self.magnitude = int(magnitude)
        self.tags = tags
        self.setTags(tags)

    def __repr__(self):
        return '<Feature>[{}: sign: {}, magnitude: {}, tags: {}]'.format(self.feature,
                                                                self.sign,
                                                                self.magnitude,
                                                                self.tags)

    def setTags(self, tags):
        self.inSentence = True if 'u' in tags else False
        self.containsSuggestion = True if 's' in tags else False
        self.containsCompetingBrand = True if 'cc' in tags else False
        self.containsSameBrand = True if 'cs' in tags else False


class TaggedSentence(object):

    def __init__(self, sentence, features=[]):
        self.features = features
        self.sentence = sentence

    def __repr__(self):
        return '<TaggedSentence>[features: {}, sentence:{}]'.format(self.features,
                                                                    self.sentence)

TAG_PAT = re.compile("\[([upsc]+)+\]")
FEATURE_PAT = re.compile("(?P<feature>.*?)\[(?P<sign>[+-])(?P<magnitude>[0-9]+)\]")

def parse_file(filepath):
    f = open(filepath)
    fr = f.read()
    f.close()

    reviews = fr.split('[t]')
    for i, review in enumerate(reviews):
        strippedReview = []
        for line in review.split('\r\n'):
            line = line.strip()
            if len(line) > 0:
                strippedReview.append(line)
        reviews[i] = strippedReview
    reviews = filter(lambda l: len(l) > 0, reviews)

    parsedReviews = []
    for rawReview in reviews:
        review = []
        for line in rawReview:
            review.append(parse_line(line))
        parsedReviews.append(review)
    return parsedReviews

def parse_line(line):
    try:
        metadata, sentence = line.split('##', 1)
        metadata, sentence = metadata.split(','), sentence.strip()
        rawFeaturesList = map(lambda c: re.search(FEATURE_PAT, c), metadata)
        tagsList = map(lambda c: re.findall(TAG_PAT, c), metadata)
        assert(len(rawFeaturesList) == len(tagsList)) # would imply shoddy regexes
        featureList = []
        for feature, tags in zip(rawFeaturesList, tagsList):
            if feature:
                featureList.append(Feature(feature.group('feature'),
                                           feature.group('sign'),
                                           feature.group('magnitude'),
                                           tags))
    except Exception as e:
        print e
        print line
    
    return TaggedSentence(sentence, featureList)

def main():
    directory = "data/training"
    os.chdir(directory)
    files = glob.glob("*.txt")
    for filename in files:
        parsedReviews = parse_file(os.path.join(os.curdir, filename))
        reviewText = ""
        for review in parsedReviews:
            for taggedSentence in review:
                reviewText += taggedSentence.sentence + '\n'
        with open(os.path.join('../clean', filename), 'w') as f:
            f.write(reviewText)

if __name__ == "__main__":
    main()


import nltk
import string


lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
table = string.maketrans("","")


stopwordsDict = {}
with open('stopwords.dict') as f:
    for word in f.readlines():
        word = word.lower().strip()
        for form in [word, lemmatizer.lemmatize(word), stemmer.stem_word(word)]:
            stopwordsDict[form] = 1


def stripPunct(s):
    return s.translate(table, string.punctuation)
    

def tokenize(s):
    return nltk.tokenize.wordpunct_tokenize(s)


def lower(word):
    return word.lower()


def lemmatize(word):
    return lemmatizer.lemmatize(word)


def stem(word):
    return stemmer.stem_word(word)


def removeStopwords(word, stopwords=stopwordsDict):
    return None if word in stopwords else word


def chainFilter(*fns):
    def f(seq):
        for fn in fns:
            seq = filter(lambda w: w if w else False, map(fn, seq))
        return seq
    return f


if __name__ == "__main__":
    words = "I am a stupid SenTence 12.".split(" ")
    filterFn = chainFilter(lower, lambda w: removeStopwords(w))
    print filterFn(words)

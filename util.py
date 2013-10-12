import os


def getWordSentimentDict():
    sentimentDict = {}
    with open('word_sentiment.dict', 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                pass
            else:
                word, sentiment = line.split('\t')
                sentimentDict[word] = int(sentiment)
    return sentimentDict


def main():
    print(getWordSentimentDict())


if __name__ == "__main__":
    main()
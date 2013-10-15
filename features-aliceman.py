from __future__ import division
import classify
import noun_phrases_extraction
import nltk
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tagger



# tagPosNeg() function tags each sentence to positve or negative base on the
# score 1 -1 0  to 'pos' 'neg' and 'neu'. 
# It also convert all reviews to lower case and get rid of punctuations
def tagPosNeg(reviews):
    tagPosNegReviews = []
    for (sent, score)in reviews:
        filtered_words = [w.lower() for w in word_tokenize(sent) if w not in stopwords.words('english')]
        tokens = [w.lower() for w in filtered_words if w not in ',.:;?/-_\|']
               
        if(score > 0):
            rate = "pos"
        elif(score < 0):
            rate = "neg"
        else:
            rate = "neu"
        tup = (tokens, rate)
        tagPosNegReviews.append(tup)
    return tagPosNegReviews


# extract only nouns
# compactness pruning(checks features that contain at least two words)
# tag 1 -1 to pos and neg
def make_noun_phrase(reviews, prune):
    sent_noun = []
    for (words, score) in reviews:
        nouns = noun_phrases_extraction.extract_noun_phrases(words)
        if(prune):
            token_nouns = [noun for noun in nouns if len(word_tokenize(noun))>=2 ]
        else:
            token_nouns = nouns

        if(score > 0):
            rate = "pos"
        elif(score < 0):
            rate = "neg"
        else:
            rate = "neu"
        sent_noun.append((token_nouns, rate))
    return sent_noun
        

# make the features set
def getwords(sent):
    justWords = []
    for(words, rate) in sent:
        justWords.extend(words)  
    return justWords
    #return feature_freqdist(justWords)
    
def feature_freqdist(lsWord):
    lsWord = nltk.FreqDist(lsWord)
    word_features = lsWord.keys()
    return word_features

def extract_feature_freqdist(sentPosNeg):
    #word_features = getwords(sentPosNeg)
    #word_features = feature_freqdist(getwords(sentPosNeg))
    sentPosNeg_words = set(sentPosNeg)
    #sentPosNeg_words = sentPosNeg
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in sentPosNeg_words)
    return features


# testing accuracy
def testingAccuracy(classifier_choice, featuresets):
    size = int(len(featuresets) * 0.1)
    train_set, test_set = featuresets[size:], featuresets[:size]

    # naive bayes
    if(classifier_choice == "naive_bayes"):
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        print "Naive-bayes accuracy: "
        print nltk.classify.accuracy(classifier, test_set)
        #(just freq) 0.7372881355932204
        #(noun-phrase freq) 0.6398305084745762

    # decision tree
    if(classifier_choice == "decision_tree"):
        classifier = nltk.DecisionTreeClassifier.train(train_set)
        print "Decision tree accuracy: "
        print nltk.classify.accuracy(classifier, test_set)

    # svm
    if(classifier_choice == "svm"):
        classifier = SklearnClassifier(LinearSVC()).train(train_set)
        print "SVM accuracy: "
        print nltk.classify.accuracy(classifier, test_set)
        # (just freq) 0.7457627118644068
    	# (noun-phrase freq)0.7245762711864406


# cross-validation
def cv (k, featuresets, classifier_choice):    
    print "%d-fold Cross Validation\t" %k
    print "Classifier: %s \n" % classifier_choice
    sum_accuracy = 0
    for i in range(0,k):
        ratio = 1 / k
        size = int(len(featuresets) * ratio)
        test_start = i * size 
        test_end = size * (i + 1)
        test_set = featuresets[test_start:test_end]
        if(i==0):
            train_start = size + 1
            train_set = featuresets[train_start:]
            #print "train index from %d  to end" % train_start
        else:
            train_end1 = test_start - 1
            train_start2 = test_end + 1
            train_set = featuresets[:train_end1] + featuresets[train_start2:]
            #print "train index_1 from 0 to %d" % train_end1
            #print "train index_2 from %d to end" %train_start2
        if(classifier_choice == 'svm'):    
            classifier = SklearnClassifier(LinearSVC()).train(train_set)
            accuracy = nltk.classify.accuracy(classifier, test_set)
            sum_accuracy += accuracy
        elif(classifier_choice == 'naive_bayes'):
            classifier = nltk.NaiveBayesClassifier.train(train_set)
            accuracy = nltk.classify.accuracy(classifier, test_set)
            sum_accuracy += accuracy
        else:
            print "Not a valid classfier name"
            return 0  
        #print "test index from %d to %d" %(test_start, test_end)
        print "Accuracy of c-v for time %d is %f \n" %(i ,accuracy)
    average_accuracy = sum_accuracy / k
    print "\nAverage accuracy is %f" % average_accuracy 
    


def main():
    reviews = classify.taggedReviews()
    #heldout_reviews = classify.heldoutReviews()
    global word_features
    
    #--------------------------------- Variation 1---------------------------------------#
    print "variation 1: Original sentences"
    print "Training:"
    sentPosNeg = tagPosNeg(reviews)
    
    word_features = feature_freqdist(getwords(sentPosNeg))
    featuresets = nltk.classify.util.apply_features(extract_feature_freqdist, sentPosNeg)
    testingAccuracy("naive_bayes", featuresets)
    testingAccuracy("svm", featuresets)
    cv(10,featuresets,"naive_bayes")
    cv(10,featuresets,"svm")
    '''
    print "Heldout:"
    sentPosNeg = tagPosNeg(heldout_reviews)
    word_features = feature_freqdist(getwords(sentPosNeg))
    featuresets = nltk.classify.util.apply_features(extract_feature_freqdist, sentPosNeg)
    testingAccuracy("naive_bayes", featuresets)
    testingAccuracy("svm", featuresets)
    cv(10,featuresets,"naive_bayes")
    cv(10,featuresets,"svm")
    '''
    #--------------------------------- Variation 2 --------------------------------------#
    print "Variation 2: Noun phrases"
    print "Training:"
    sentPosNeg_nouns = make_noun_phrase(reviews, False)
    word_features = feature_freqdist(getwords(sentPosNeg_nouns))
    featuresets = nltk.classify.util.apply_features(extract_feature_freqdist, sentPosNeg_nouns)
    testingAccuracy("naive_bayes", featuresets)
    testingAccuracy("svm", featuresets)
    cv(10,featuresets,"naive_bayes")
    cv(10,featuresets,"svm")
    '''
    print "Heldout:"
    sentPosNeg = tagPosNeg(heldout_reviews)
    word_features = feature_freqdist(getwords(sentPosNeg))
    featuresets = nltk.classify.util.apply_features(extract_feature_freqdist, sentPosNeg)
    testingAccuracy("naive_bayes", featuresets)
    testingAccuracy("svm", featuresets)
    cv(10,featuresets,"naive_bayes")
    cv(10,featuresets,"svm")
    '''
    # --------------------------------- Variation 3 --------------------------------------#
    print "Variation 3: Noun Phrases with pruning"
    print "Training:"
    sentPosNeg_nouns_prune = make_noun_phrase(reviews, True)
    word_features = feature_freqdist(getwords(sentPosNeg_nouns_prune))
    featuresets = nltk.classify.util.apply_features(extract_feature_freqdist, sentPosNeg_nouns_prune)
    testingAccuracy("naive_bayes", featuresets)
    testingAccuracy("svm", featuresets)
    cv(10,featuresets,"naive_bayes")
    cv(10,featuresets,"svm")
    '''
    print "Heldout:"
    sentPosNeg = tagPosNeg(heldout_reviews)
    word_features = feature_freqdist(getwords(sentPosNeg))
    featuresets = nltk.classify.util.apply_features(extract_feature_freqdist, sentPosNeg)
    testingAccuracy("naive_bayes", featuresets)
    testingAccuracy("svm", featuresets)
    cv(10,featuresets,"naive_bayes")
    cv(10,featuresets,"svm")
    '''
    
    

if __name__ == '__main__':
    main()

               

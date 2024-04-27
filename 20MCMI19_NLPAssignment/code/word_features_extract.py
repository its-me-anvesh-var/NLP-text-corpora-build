import matplotlib.pyplot as plt
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import PlaintextCorpusReader

corpusdir = "/home/anvesh/Desktop/20MCMI19_NLPAssignment/new corpus"


#word featuring


newcorpus = PlaintextCorpusReader(corpusdir, '.*')
stop_words = set(stopwords.words('english') + list(punctuation))


all_words = []
words = newcorpus.words()
for w in words:
    if w not in stop_words:
        all_words.append(w.lower())

freqdist = nltk.FreqDist(all_words)
word_features_set = list(freqdist.keys())[:3000]

def find_features_set(filename):
    words = set(filename)
    features_set = {}
    for w in word_features_set:
        features_set[w] = (w in words)

    return features_set

print((find_features_set(newcorpus.words('text_news.txt'))))



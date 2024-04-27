import nltk
from string import punctuation
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import PlaintextCorpusReader


corpusdir = "/home/anvesh/Desktop/20MCMI19_NLPAssignment/new corpus"



stop_words = set(stopwords.words('english') + list(punctuation))


data = open("/home/anvesh/Desktop/20MCMI19_NLPAssignment/new corpus/text_news.txt", 'r')


output  = [word for word in nltk.word_tokenize(data.read()) if word.lower() not in stop_words]

print(nltk.FreqDist(ngrams(output,5)).most_common(20))

import matplotlib.pyplot as pt
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import PlaintextCorpusReader

corpusdir = "/home/anvesh/Desktop/20MCMI19_NLPAssignment/new corpus"


newcorpus = PlaintextCorpusReader(corpusdir, '.*')
stop_words = set(stopwords.words('english') + list(punctuation))


#word frequency analysis and tabulating 20 words


all_words = []
words = newcorpus.words()
for w in words:
    if w not in stop_words:
        all_words.append(w.lower())
        
    

freqdist = nltk.FreqDist(all_words)
print(freqdist.tabulate(20))
freqdist.plot(20,cumulative = True)




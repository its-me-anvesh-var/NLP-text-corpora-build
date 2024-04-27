import matplotlib.pyplot as pt
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import PlaintextCorpusReader

corpusdir = "/home/anvesh/Desktop/20MCMI19_NLPAssignment/new corpus"

#chinking

training_text = open("/home/anvesh/Desktop/20MCMI19_NLPAssignment/new corpus/text_news.txt", 'r')
testing_text = open("/home/anvesh/Desktop/20MCMI19_NLPAssignment/new corpus/text_mag.txt", 'r')

sample_sentence_tokenizer = PunktSentenceTokenizer(training_text.read())

tokenized_one = sample_sentence_tokenizer.tokenize(testing_text.read())

def process_content():
    try:
        for i in tokenized_one[:5]:
            words = nltk.word_tokenize(i)
            tagged_one = nltk.pos_tag(words)
            chunking = r"""Chunk: {<.*>+}
                                        }<VB.?|IN|DT|TO>+{"""
            chunkParse = nltk.RegexpParser(chunking)
            chunked = chunkParse.parse(tagged_one)
            chunked.draw()

            
            print(tagged_one)

    except Exception as e:
        print(str(e))

process_content()

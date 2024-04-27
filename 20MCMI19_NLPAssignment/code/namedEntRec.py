import nltk
from nltk.tokenize import word_tokenize,sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import PlaintextCorpusReader

corpusdir = "/home/anvesh/Desktop/20MCMI19_NLPAssignment/new corpus"

training_text = open("/home/anvesh/Desktop/20MCMI19_NLPAssignment/new corpus/text_mag.txt", 'r')
testing_text = open("/home/anvesh/Desktop/20MCMI19_NLPAssignment/new corpus/text_news.txt", 'r')

sample_tokenizer = PunktSentenceTokenizer(training_text.read())

tokenized_data = sample_tokenizer.tokenize(testing_text.read())

#named entity recognition

def process_content():
    try:
        for i in tokenized_data[:5]:
            words_tokenized = nltk.word_tokenize(i)
            tagged_one = nltk.pos_tag(words_tokenized)
            namedEntityRec = nltk.ne_chunk(tagged_one, binary = False)
            namedEntityRec.draw()

    except Exception as e:
        print(str(e))

process_content()

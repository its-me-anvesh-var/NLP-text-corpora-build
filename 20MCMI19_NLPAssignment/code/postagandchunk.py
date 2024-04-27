import nltk
from nltk.tokenize import word_tokenize,sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import PlaintextCorpusReader

corpusdir = "/home/anvesh/Desktop/20MCMI19_NLPAssignment/new corpus"

#pos tagging and chunking

training_text = open("/home/anvesh/Desktop/20MCMI19_NLPAssignment/new corpus/text_acad.txt", 'r')
testing_text = open("/home/anvesh/Desktop/20MCMI19_NLPAssignment/new corpus/text_blog.txt", 'r')

sample_sentence_tokenizer = PunktSentenceTokenizer(training_text.read())

tokenized_one = sample_sentence_tokenizer.tokenize(testing_text.read())

def process_content():
    try:
        for i in tokenized_one[:5]:
            words = nltk.word_tokenize(i)
            tagged_one = nltk.pos_tag(words)
            chunking = r"""Chunk: {<RB.?><VB.?><NNP>+<NN>?}"""
            chunkParse = nltk.RegexpParser(chunking)
            chunked = chunkParse.parse(tagged_one)
            chunked.draw()

            
            print(tagged_one)

    except Exception as e:
        print(str(e))

process_content()

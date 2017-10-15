import os 
import spacy
from gensim import corpora, models, similarities
import gensim
import numpy as np


chapter_num  =   4  #initialize parameter 



def read_data():
    #import stoplist
    stoplist = []
    with open('stoplist.txt') as f:
        for word in f.readlines():
            stoplist.append(word[:-1])
            
    #import scripts data, output plain text
    path     =  '/Users/zhanzhuoyang/Desktop/Columbia University/MSAA Curriculum/2017 Fall/Captsone/scripts/'
    namelist =  os.listdir(path)
    docs     =  []
    for name in namelist:
        if name[0] != '.':
            doc = []
            with open(path + name) as f:
                a = f.readlines()
                for line in a[1:]:
                    if (line[0].isdigit() == False) and (line != '\n'):
                        if (line[0] == '-'):
                            doc.append(line[1:-1])
                        elif (line[1] == '-'):
                            doc.append(line[2:-1])
                        else:
                            doc.append(line[:-1])
                docs.append(doc)
    return docs, stoplist


def bag_of_words(docs, stoplist):    #cut every episode into 4 same length chapter
    texts = []
    for doc in docs:
        start =   0
        seg   =   int(len(doc)/chapter_num)
        for time in range(chapter_num):
            chapter =   doc[start: (start + seg)]
            text    =   []
            for line in chapter:
                b = line.split()
                for word in b:
                    if word[-1] in {'.',',','!','?'}:
                        word = word[:-1]
                    if word.lower() not in stoplist:
                        text.append(word.lower())
            start += seg
            texts.append(text)
            
    return texts

def extract_keywords(texts):     #build up dictionary and calculate the tfidf value of each words
    dictionary        =    corpora.Dictionary(texts)
    corpus            =    [dictionary.doc2bow(text) for text in texts]
    tfidf_model       =    models.TfidfModel(corpus) 
    corpus_tfidf_list =    tfidf_model[corpus]
    index = 0
    print('keywords for each episode: ')
    for corpus_tfidf in corpus_tfidf_list:
        keywords = []
        keywords_tfidf = sorted(corpus_tfidf, key=lambda corpus_tfidf: corpus_tfidf[1], reverse = True) [:11]
        [keywords.append(dictionary[i[0]]) for i in keywords_tfidf]
        if (index % chapter_num) == 0:
            print('\n','Episode', str(int((index/chapter_num+1))))
        print(keywords)
        index += 1

def word2vec(texts):
    model = gensim.models.word2vec.Word2Vec(texts, size=100, window=5, min_count=5, workers=4)
    return model
# word_vector = model[word]


docs, stoplist = read_data()
texts = bag_of_words(docs, stoplist)
extract_keywords(texts)
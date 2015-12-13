# -*- coding: utf-8 -*-

from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
import sys

num_topics = 20

vect = TfidfVectorizer()   # конвертор в матрицу TF-IDF
tok = vect.build_tokenizer()  # токенизатор

lda = models.LdaModel.load('lda.dat')
dictionary = corpora.Dictionary.load('dict.dat')

# ПОСТРОЕНИЕ МАТРИЦЫ
row = lda.show_topic(topicid=0, topn=len(dictionary))
row = sorted(row, key=lambda (word, p): word)
alphabetOrderedDict = []
matrix = []
for tup in row:
    alphabetOrderedDict.append(tup[0])
    matrix.append([tup[1]])
for i in xrange(1,num_topics):
    row = lda.show_topic(topicid=i, topn=len(dictionary))
    row = sorted(row, key=lambda (word, p): word)
    for j,tup in enumerate(row):
        matrix[j].append(tup[1])

# НОВОЕ СЛОВО
new_vec = dictionary.doc2bow((tok('gun')))
new_word=[]
for tup in lda[new_vec]:
    new_word.append(tup[1])

finalList = []
for i in xrange(0, len(dictionary)):
    result = spatial.distance.cosine(new_word, matrix[i])
    finalList.append((alphabetOrderedDict[i],result))
finalList = sorted(finalList, key=lambda (word, cos): cos)
print finalList[:10]

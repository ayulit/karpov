# -*- coding: utf-8 -*-

from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy import spatial

import numpy as np
import sys



vect = TfidfVectorizer()   # конвертор в матрицу TF-IDF
tok = vect.build_tokenizer()  # токенизатор
lem = WordNetLemmatizer() # лемматизатор

dataset=fetch_20newsgroups()  # датасет - 20 групп новостей




################################################################################


# remove common words, tokenize and lemmatize
#texts = [[lem.lemmatize(word) for word in tok(text.lower()) if word not in stopwords.words('english')]
#         for text in dataset.data]

# remove words that appear only once
#from collections import defaultdict
#frequency = defaultdict(int)
#for text in texts:
#    for token in text:
#        frequency[token] += 1
#texts = [[token for token in text if frequency[token] > 1]
#         for text in texts]

#dictionary = corpora.Dictionary(texts) # создаем словарь (сет токенов)
#dictionary.save("dict.dat")
#corpus = [dictionary.doc2bow(text) for text in texts] # корпус

num_topics = 100

# Обучение LDA модели
#lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, update_every=1, chunksize=10000, passes=1)
#lda.save('lda.dat')

################################################################################

lda = models.LdaModel.load('lda.dat')
dictionary = corpora.Dictionary.load('dict.dat')



# ПОСТРОЕНИЕ МАТРИЦЫ

row = lda.show_topic(topicid=0, topn=len(dictionary))
row = sorted(row, key=lambda (p, word): word)

alphabetOrderedDict = []
matrix = []
for tup in row:
    alphabetOrderedDict.append(tup[1])
    matrix.append([tup[0]])


for i in xrange(1,num_topics):
    row = lda.show_topic(topicid=i, topn=len(dictionary))
    row = sorted(row, key=lambda (p, word): word)
    for j,tup in enumerate(row):
       matrix[j].append(tup[0])

new_vec = dictionary.doc2bow((tok('world')))  # это нигде не используется
new_word=[]
for tup in lda[new_vec]:
    new_word.append(tup[1])

finalList = []
for i in xrange(0, len(dictionary)):
    result = spatial.distance.cosine(new_word, matrix[i])
    finalList.append((alphabetOrderedDict[i],result))
finalList = sorted(finalList, key=lambda (word, cos): cos)
print finalList



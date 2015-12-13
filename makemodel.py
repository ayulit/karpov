# -*- coding: utf-8 -*-

from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re

import numpy as np
import sys

num_topics = 20

vect = TfidfVectorizer()   # конвертор в матрицу TF-IDF
tok = vect.build_tokenizer()  # токенизатор
lem = WordNetLemmatizer() # лемматизатор

dataset=fetch_20newsgroups()  # датасет - 20 групп новостей

# berem toko 3 categorii
# dataset=fetch_20newsgroups(categories=['alt.atheism', 'talk.religion.misc', 'sci.space'])



#########################################################################################

# remove common words, tokenize and lemmatize
texts = [[lem.lemmatize(word) for word in tok(text.lower())\
          if ((word not in stopwords.words('english'))\
              and (word.isdigit() == False)\
              and not re.search("\d", word)\
              and not re.search("_", word))]
         for text in dataset.data]

print "texts - ok"

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

dictionary = corpora.Dictionary(texts) # создаем словарь (сет токенов)
dictionary.save("dict.dat")

corpus = [dictionary.doc2bow(text) for text in texts] # корпус

# Обучение LDA модели
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, update_every=1, chunksize=10000, passes=1)
lda.save('lda.dat')
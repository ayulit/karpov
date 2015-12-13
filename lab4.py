__author__ = '315-4'
# -*- coding: utf-8 -*-
from gensim import corpora, models, matutils
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

dataset=fetch_20newsgroups()  # датасет - 20 групп новостей

vect = TfidfVectorizer()  # конвертор в матрицу TF-IDF
tok = vect.build_tokenizer()  # токенизатор

texts = []
# токенизация текстов
for text in dataset.data:
    texts.append(tok(text))

# на сцену выходит gensim
# Convert document (a list of words) into the bag-of-words
dictionary = corpora.Dictionary(texts)  # создаем словарь (сет токенов)
corpus = [dictionary.doc2bow(text) for text in texts]  # корпус

new_vec = dictionary.doc2bow((tok('Hello world')))  # это нигде не используется

# Обучение LDA модели
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,\
                               num_topics=100, update_every=1, chunksize=10000, passes=1)

# выводим матрицу V из UEV разложения
for item in lda.print_topics(100):
    print (item)
    
# Функция распределения топиков внутри докуметов
# документ 1 в нашем случае - и мы увидим там какие содержаться топики по словам
print lda[new_vec]







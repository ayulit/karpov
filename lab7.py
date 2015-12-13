# -*- coding: utf-8 -*-
from gensim import corpora, models, matutils
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

dataset=fetch_20newsgroups(categories=['alt.atheism','talk.religion.misc','sci.space']) # berem toko 3 categorii
vect = TfidfVectorizer()
tok=vect.build_tokenizer() # хорошо токенизирует все
texts=[]
lem=WordNetLemmatizer()
lemms=[]
#for text in dataset.data:
#    for token in tok(text):
#        lemms.append(lem.lemmatize(token))
#    texts.append(lemms)
#models = models.Word2Vec(texts,size=100, window=5,min_count=5,workers=4)
#models.save('texts.dat')

model = models.Word2Vec.load('texts.dat')
#print(model['theory'])
#print(model.similarity('man','car'))
#print(model.most_similar(positive=['man'],negative=['computer']))
print model.doesnt_match("car wheel glass engine".split())
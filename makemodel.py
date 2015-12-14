# -*- coding: utf-8 -*-

from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

num_topics = 20 # number of topics

vect = TfidfVectorizer()
tok = vect.build_tokenizer()  # tokenizer
lem = WordNetLemmatizer()  # lemmatizer

dataset=fetch_20newsgroups()  # dataset

# remove common words, numbers, tokenize and lemmatize
texts = [[lem.lemmatize(word) for word in tok(text.lower())\
          if ((word not in stopwords.words('english'))\
              and word.isalpha())]
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

dictionary = corpora.Dictionary(texts) # dictionary
dictionary.save("dict.dat")  # saving dictionary

corpus = [dictionary.doc2bow(text) for text in texts]  # corpus

# LDA model
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, update_every=1, chunksize=10000, passes=1)
lda.save('lda.dat')  # saving model
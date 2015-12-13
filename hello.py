# -*- coding: utf-8 -*-

from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

vect = TfidfVectorizer()   # конвертор в матрицу TF-IDF
tok = vect.build_tokenizer()  # токенизатор
lem = WordNetLemmatizer()  # лемматизатор

data = ["mom washed window", "mom asked her mom"]
texts = []
for text in data:
    tokens = tok(text)
    filtered_words = [lem.lemmatize(token.lower())\
                      for token in tokens if token.lower() not in stopwords.words('english')]
    texts.append(filtered_words)


dictionary = corpora.Dictionary(texts) # создаем словарь (сет токенов)

print dictionary

print ["foo","bar"].index("foo")

print "hellpopo1231231".isdigit()

word = "super1"
if re.search("\d", word):
    print("y")
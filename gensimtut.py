# -*- coding: utf-8 -*-

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

vect = TfidfVectorizer()   # конвертор в матрицу TF-IDF
tok = vect.build_tokenizer()  # токенизатор
lem = WordNetLemmatizer()  # лемматизатор

dataset = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# remove common words, tokenize and lemmatize
texts = [[lem.lemmatize(word) for word in tok(text.lower()) if word not in stopwords.words('english')]
         for text in dataset]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

print texts
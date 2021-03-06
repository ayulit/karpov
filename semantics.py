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
row1 = lda.show_topic(topicid=0, topn=len(dictionary))
row2 = sorted(row1, key=lambda (p, word): word)
alphabetOrderedDict = []
matrix = []
for tup in row2:
    alphabetOrderedDict.append(tup[1])
    matrix.append([tup[0]])
for i in xrange(1,num_topics):
    row1 = lda.show_topic(topicid=i, topn=len(dictionary))
    row2 = sorted(row1, key=lambda (p, word): word)
    for j,tup in enumerate(row2):
        matrix[j].append(tup[0])

# НОВОЕ СЛОВО
new_vec = dictionary.doc2bow((tok('gun')))

#print lda[new_vec][:5]
#sys.exit("stop")

new_word=[]
for tup in lda[new_vec]:
    new_word.append(tup[1])

finalList = []
for i in xrange(len(dictionary)):
    result = spatial.distance.euclidean(new_word, matrix[i])
    finalList.append((alphabetOrderedDict[i],result))
finalList2 = sorted(finalList, key=lambda (word, cos): cos)
print finalList2[:10]

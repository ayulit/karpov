# -*- coding: utf-8 -*-

from gensim import corpora, models
from scipy import spatial

# New word
word = "freedom"
num_topics = 20 # number of topics

# load model
lda = models.LdaModel.load('lda.dat')
dictionary = corpora.Dictionary.load('dict.dat')

# Matrix construction
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

# Get top-10
if word in alphabetOrderedDict:
    word_ind = alphabetOrderedDict.index(word)
finalList = [] # top
for i in xrange(len(dictionary)):
    result = spatial.distance.cosine(matrix[word_ind], matrix[i])
    finalList.append((alphabetOrderedDict[i],result))
finalList2 = sorted(finalList, key=lambda (word, cos): cos)

# print top 10
for tup in finalList2[:10]:
    print tup[0]


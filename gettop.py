from gensim import corpora, models
from scipy import spatial

import sys

# sort tuples in a row depend on their format (gensim version)
def sortDependOnGensim(row):
    if type(row[0][1]) == unicode:
        # old gensim (p, word)
        row2 = sorted(row, key=lambda (p, word): word)
        ver = "old"
    else:
        # new gensim (word, p)
        row2 = sorted(row, key=lambda (word, p): word)
        ver = "new"
    return row2, ver


def indexesDependOnGensim(ver):
    if ver == "old":
        # old gensim
        p_ind, w_ind = 0, 1
    else:
        p_ind, w_ind = 1, 0
    return p_ind, w_ind

# New word
word = "computer"
num_topics = 20 # number of topics

# load model
lda = models.LdaModel.load('lda.dat')
dictionary = corpora.Dictionary.load('dict.dat')

# Matrix construction
row1 = lda.show_topic(topicid=0, topn=len(dictionary))
row2, ver = sortDependOnGensim(row1)
p_ind, w_ind = indexesDependOnGensim(ver)

alphabetOrderedDict = []
matrix = []
for tup in row2:
    alphabetOrderedDict.append(tup[w_ind])
    matrix.append([tup[p_ind]])
for i in xrange(1,num_topics):
    row1 = lda.show_topic(topicid=i, topn=len(dictionary))
    row2, ver = sortDependOnGensim(row1)
    for j,tup in enumerate(row2):
       matrix[j].append(tup[p_ind])

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


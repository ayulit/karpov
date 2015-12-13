__author__ = '315-4'

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans# zaimpotrim k-means
import sklearn.metrics as metrics # zaimoptrim libu po metrikam ka4estva

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
vect = TfidfVectorizer()

#import nltk
#nltk.download()

def tokenize(text):
    stems=[]
    lems=[]
    tok=vect.build_tokenizer()
    tokens=tok(text)
    st=PorterStemmer()
    lem=WordNetLemmatizer()
    for token in tokens:
        stems.append(st.stem(token))
        lems.append(lem.lemmatize(token))

    return lems


tfidf_ngrams=TfidfVectorizer(tokenizer=tokenize)
dataset=fetch_20newsgroups(categories=['alt.atheism','talk.religion.misc','sci.space']) # berem toko 3 categorii

print(len(dataset.data)) # spisok documentov
labels = dataset.target
print(dataset.target_names) # skisok kategoriy
print(len(dataset.target_names))
X=tfidf_ngrams.fit_transform(dataset.data)
#print(X) # vyvod razrez matricy s chastotoy

# sdelaem klasterizaciu kmeans
km=KMeans(n_clusters=3) # sozdaly
km.fit(X) # obuchaem

# s4itaem ka4estvo klasterizacii po koli4estvu tekstov v klastere

print labels,km.labels_
print metrics.homogeneity_score(labels,km.labels_) # toko owibka 1-go roda
print metrics.completeness_score(labels,km.labels_) #

# mozem pos4itat lemmatizaciu

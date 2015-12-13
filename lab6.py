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
X=tfidf_ngrams.fit_transform(dataset.data)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
vsd=TruncatedSVD(n_components=3)
normalizer=Normalizer(copy=False)
lsa=make_pipeline(vsd,normalizer)
d_t=lsa.fit_transform(X)
print d_t



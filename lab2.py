from nltk import probability
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
corpus = [
'The advantages of support vector machines are',
'Effective in high dimensional spaces',
'Still effective in cases where number of dimensions is greater than the number of samples',
'Uses a subset of training points in the decision function called support vectors so it is also memory efficient',
'Versatile different Kernel functions can be specified for the decision function Common kernels are provided but it is also possible to specify custom kernels',
'The acting head of the Russian Athletics Federation Vadim Zelichenok said there were few fresh facts in the report and past problems with doping had been tackled',
'The report depicted a culture of systematic cheating with even the secret services involved',
'It said neither the All-Russia Athletics Federation Araf the Russian anti-doping agency Rusada nor the Russian Athletics Federation were complying with anti-doping procedures',
'The report by an independent commission for the World Anti-Doping Agency Wada sent shockwaves through the world of sport',
'Australia and the UK have backed its call to ban Russia from all competitions including next year Olympics in Rio de Janeiro']

# learning vectorizer by corpus
X = vectorizer.fit_transform(corpus).toarray()

# create tokenizator depends on vectorizer
analyze = vectorizer.build_analyzer()
new_text = " Russian Athletics Federation"
new_vector=vectorizer.transform([new_text]).toarray()

# lab 2
# let's use classifiers
from sklearn import svm

# razmetim 0 - tex, 1 - bbc
y = [0,0,0,0,0,1,1,1,1,1]
clf=svm.SVC(probability=True, kernel = 'linear')
clf.fit(X,y)

print clf.predict_proba(new_vector)
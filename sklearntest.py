from sklearn.datasets import fetch_20newsgroups

dataset=fetch_20newsgroups(categories=['alt.atheism'])

print(len(dataset.data)) # spisok documentov
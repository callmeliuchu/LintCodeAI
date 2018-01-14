import csv
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
inputfile = '../data/train.csv'




def getData():
	csv_reader = csv.reader(open(inputfile,encoding='utf-8'))
	data = np.array(list(csv_reader)[1:])
	return data


def convert(label):
	return 0 if label == 'ham' else 1


def getLabelsAndContens():
	data = getData()
	labels = [convert(arr[0]) for arr in data]
	contents = [arr[1] for arr in data]
	return labels,contents



def review_to_wordlist(review,remove_stopwords=False):
	review_text = re.sub("[^a-zA-Z]"," ",review)
	words = review_text.lower().split()
	if remove_stopwords:
		words = [word for word in words if word not in stopwords.words('english')]
	return words

def dealWithContent(review):
	review_text = re.sub("[^a-zA-Z]"," ",review)
	return review_text


from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem
english_stemmer = nltk.stem.SnowballStemmer('english')


class StemmedTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer = super(TfidfVectorizer,self).build_analyzer()
		return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))


labels,contents = getLabelsAndContens()
# from sklearn.feature_extraction.text import CountVectorizer
vectorizer = StemmedTfidfVectorizer(min_df=1,stop_words='english')
contents = [dealWithContent(sentence) for sentence in contents]
train = vectorizer.fit_transform(contents)
print(train.shape)
print(vectorizer.get_feature_names())
# new_post = "bitching attended"
# new_post_vec = vectorizer.transform([new_post])
# print(new_post_vec)
def getConvertData(contents,vectorizer):
	data_set = []
	for sentence in contents:
		new_sentence = vectorizer.transform([sentence])
		data_set.append(new_sentence.toarray()[0])
		# print(new_sentence.toarray()[0])
		# print("-----------------------------")
	return np.array(data_set)


from sklearn.decomposition import PCA
pca = PCA(n_components=10)
data_set = getConvertData(contents,vectorizer)
new_data = pca.fit_transform(data_set)
print(new_data)








# def getSentences():
# 	labels,contents = getLabelsAndContens()
# 	sentences = [sentence for sentence in contents]

# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# s = contents[0]
# print(tokenizer.tokenize(s))
# labels,contents = getLabelsAndContens()
# for sentence in contents:
# 	print(review_to_wordlist(sentence,True))
# from gensim.models import word2vec
# num_features = 300
# min_word_count = 40
# num_workers = 4
# context = 10
# downsampling = 1e-3

# model = word2vec.Word2Vec(contents,workers=num_workers,\
# 	size=num_features,min_count=min_word_count,\
# 	window=context,sample=downsampling)
# model.init_sims(replace=True)
# model.save('data')
# from gensim.models import Word2Vec
# model = Word2Vec.load("data")
# print(set(model.index2word))
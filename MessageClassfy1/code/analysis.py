import csv
import numpy as np
import pandas as pd
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
import bayes
inputfile = '../data/train.csv'
testfile = '../data/test.csv'



def getData(path):
	csv_reader = csv.reader(open(path,encoding='utf-8'))
	data = np.array(list(csv_reader)[1:])
	return data


def convert(label):
	return 0 if label == 'ham' else 1


def getTestIdsAndContents(path):
	data = getData(path)
	labels = [arr[0] for arr in data]
	contents = [arr[1] for arr in data]
	return labels,contents

def getLabelsAndContens(path):
	data = getData(path)
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



labels,contents = getLabelsAndContens(inputfile)


ids,test_contents = getTestIdsAndContents(testfile)
# print(ids,test_contents)
# from sklearn.feature_extraction.text import CountVectorizer
vectorizer = StemmedTfidfVectorizer(min_df=1,stop_words='english')
contents = [dealWithContent(sentence) for sentence in contents]

def getSimpleDataSet(contents):
	dataSet = []
	for sentence in contents:
		dataSet.append(sentence.split())
	return dataSet

dataSet = getSimpleDataSet(contents)
# bayes = bayse_1.Bayes(dataSet,labels)


train = vectorizer.fit_transform(contents)
# print(train.shape)
# print(vectorizer.get_feature_names())
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






labesl_des = ['ham','spam']
data_set = getConvertData(contents,vectorizer)
test_data_set = getConvertData(test_contents,vectorizer)
print(test_data_set)
data_set = data_set*10
p0,p1,pa = bayes.trainNB0(data_set,labels)
print(p0)
print(p1)
print(pa)
res_arr = []
for i in range(len(ids)):
	aid = ids[i]
	test_data = test_data_set[i]
	res = bayes.classfy(test_data,p0,p1,pa)
	res_arr.append([aid,labesl_des[res]])
	print(aid,labesl_des[res])

# data_frame = pd.DataFrame(res_arr,columns=['SmsId','Label'],index=None)
# data_frame.to_csv('submission1.csv')
# print(data_frame.values)






# from sklearn.decomposition import PCA
# pca = PCA(n_components=10)
# data_set = getConvertData(contents,vectorizer)
# new_data = pca.fit_transform(data_set)
# print(new_data)








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
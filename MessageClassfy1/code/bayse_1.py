import numpy as np
from collections import Counter

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec


class Bayes:
	def __init__(self,dataSet,labels):
		self.dataSet = dataSet
		self.labels = labels

	def vocabulist(self,dataSet):
		ret = set([])
		for arr in self.dataSet:
			ret = ret | set(arr)
		return list(ret)

	def words2vec(self,words):
		vocabulist = self.vocabulist(self.dataSet)
		res = np.ones(len(vocabulist))
		for word in words:
			if word in vocabulist:
					res[vocabulist.index(word)] += 1
		return res

	def data_matrix(self):
		res = []
		for arr in self.dataSet:
			words_vec = self.words2vec(arr)
			res.append(words_vec)
		return np.array(res)

	def calculation(self):
		data_matrix = self.data_matrix()
		label_map = dict()
		counts = Counter(self.labels)
		num_words = len(self.labels)
		for i in range(len(self.labels)):
			label = self.labels[i]
			if label not in label_map:
				label_map[label] = data_matrix[i]
			else:
				label_map[label] += data_matrix[i]
		res = dict()
		for label in set(self.labels):
			res[label] = (np.log(label_map[label]/float(sum(label_map[label]))),counts[label]/float(num_words))
		return res

	def classify(self,vec):
		cal = self.calculation()
		max_val = np.inf*(-1)
		for label in set(self.labels):
			val = sum(cal[label][0]*vec)+np.log(cal[label][1])
			if max_val < val:
				max_val = val
				res_loc = label
		return res_loc

	def classifySentence(self,sentence):
		words = sentence.split()
		vec = self.words2vec(words)
		return self.classify(vec)

			



dataSet,labels = loadDataSet()
bay = Bayes(dataSet,labels)
res =bay.classifySentence("I think this is so stupid")
print(res)








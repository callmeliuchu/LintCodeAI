import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabulist(dataSet):
	ret = set()
	for data in dataSet:
		ret = ret | set(data)
	return list(ret)

def words2Vec(vocabulist,input):
	ret = [0]*(len(vocabulist))
	for word in input:
		if word in vocabulist:
			ret[vocabulist.index(word)] = 1
		else:
			print('not in this list')
	return ret




dataSet,labels = loadDataSet()
vocabulist = createVocabulist(dataSet)
for inputData in dataSet:
	print(words2Vec(vocabulist,inputData))
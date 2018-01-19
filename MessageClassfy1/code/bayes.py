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
			ret[vocabulist.index(word)] += 1
		else:
			print('not in this list')
	return ret



def trainNB0(trainMatrix,trainCategory):
	num_word = len(trainMatrix[0])
	p1vec = np.ones(num_word)
	p0vec = np.ones(num_word)
	pabusive = sum(trainCategory)/float(len(trainMatrix))
	for i in range(len(trainMatrix)):
		if trainCategory[i] == 1:
			p1vec += trainMatrix[i]
		else:
			p0vec += trainMatrix[i]
	p0 = np.log(p0vec/sum(p0vec))
	p1 = np.log(p1vec/sum(p1vec))
	return p0,p1,pabusive



def classfy(vec,p0,p1,pc1):
	p1 = sum(vec*p1) + np.log(pc1)
	p0 = sum(vec*p0) + np.log(1-pc1)
	if p1>p0:
		return 1
	else:
		return 0

dataSet,labels = loadDataSet()
vocabulist = createVocabulist(dataSet)
dataMatrix = []
for inputData in dataSet:
	dataMatrix.append(words2Vec(vocabulist,inputData))

# print(dataMatrix)
p0,p1,pa = trainNB0(np.array(dataMatrix),np.array(labels))
mysent = "fucking stupid boy"
words = mysent.split()
# print(words)
vec = words2Vec(vocabulist,words)
print(classfy(vec,p0,p1,pa))

from collections import Counter
import math
def createDataSet():
	dataSet = [[1,1,'yes'],
	          [1,1,'yes'],
	          [1,0,'no'],
	          [0,1,'no'],
	          [0,1,'no']]
	labels = ['no surfacing','flippers']
	return dataSet,labels


def calShano(dataSet):
	feats = [vec[-1] for vec in dataSet]
	num = len(feats)
	count = Counter(feats)
	return sum((-1)*(count[key]/num)*math.log((count[key]/num),2) for key in count)


def splitDataSet(dataSet,axis,value):
	res = []
	for vec in dataSet:
		if vec[axis] == value:
			arr = list(vec[:axis])
			arr.extend(list(vec[axis+1:]))
			res.append(arr)
	return res

def chooseBestFeat(dataSet):
	featNum = len(dataSet[0])-1
	vecNum = len(dataSet)
	max_gain = -1
	best_feat = -1
	baseShano = calShano(dataSet)
	for i in range(featNum):
		values = set([vec[i] for vec in dataSet])
		p = 0
		for val in values:
			data = splitDataSet(dataSet,i,val)
			shano = calShano(data)
			p += len(data)/vecNum*shano
		gainShano = baseShano - p
		if max_gain < gainShano:
			max_gain = gainShano
			best_feat = i
	return best_feat


def createTree(dataSet,labels):
	if len(dataSet[0]) == 1:
		return Counter([vec[-1] for vec in dataSet]).most_common(1)[0][0]
	if len(set([vec[-1] for vec in dataSet])) == 1:
		return dataSet[0][-1]
	best_feat = chooseBestFeat(dataSet)
	label = labels[best_feat]
	myTree = {label:{}}
	values = set([vec[best_feat] for vec in dataSet])
	for value in values:
		new_labels = labels[:best_feat] + labels[best_feat+1:]
		myTree[label][value] = createTree(splitDataSet(dataSet,best_feat,value),new_labels)
	return myTree

def classify(inputTree,labels,vec):
	firstStr = list(inputTree.keys())[0]
	secondDict = inputTree[firstStr]
	featNum = labels.index(firstStr)
	for val in secondDict:
		if val == featNum:
			if type(secondDict[val]).__name__ == 'dict':
				classLabel = classify(secondDict[val],labels,vec)
			else:
				classLabel = secondDict[val]
	return classLabel

dataSet,labels = createDataSet()
tree = createTree(dataSet,labels)
print(tree)
res = classify(tree,labels,[1,1])
print(res)
# print(dataSet)
# print(labels)
# print(calShano(dataSet))
# data = splitDataSet(dataSet,0,1)
# print(data)
# print(chooseBestFeat(dataSet))
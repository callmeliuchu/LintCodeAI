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

def calShanno(dataSet):
	labels = [vec[-1] for vec in dataSet]
	count = Counter(labels)
	num = len(labels)
	return sum((count[key]/float(num))*math.log(count[key]/float(num),2)*(-1) for key in count)

def splitDataSet(dataSet,axis,value):
	ret = []
	for vec in dataSet:
		if vec[axis] == value:
			ret.append([vec[i] for i in range(len(vec)) if i!=axis])
	return ret


def chooseBestSplit(dataSet):
	numFeature = len(dataSet[0]) - 1
	baseInfo = calShanno(dataSet)
	maxGain = -1
	bestF = -1
	for i in range(numFeature):
		features = [vec[i] for vec in dataSet]
		featuresSet  = set(features)
		info = 0.0
		for f in featuresSet:
			data = splitDataSet(dataSet,i,f)
			info += len(data)/len(features)*calShanno(data)
		gain = baseInfo - info
		if maxGain < gain:
			maxGain = gain
			bestF = i
	return bestF


def createTree(dataSet,labels):
	backValues = [vec[-1] for vec in dataSet]
	if len(set(backValues)) == 1:
		return backValues[0]
	if len(dataSet[0]) == 1:
		return Counter(backValues).most_common(1)[0][0]
	bestSplit = chooseBestSplit(dataSet)
	label = labels[bestSplit]
	myTree = {label:{}}
	del labels[bestSplit]
	features = [vec[bestSplit] for vec in dataSet]
	featureSet = set(features)
	for value in featureSet:
		data = splitDataSet(dataSet,bestSplit,value)
		newLables = labels[:]
		myTree[label][value] = createTree(data,newLables)
	return myTree


# a = [1,1,1,1,2,3,3,3,3,3,3]
# val = Counter(a).most_common(1)[0][0]
# print(val)









# dataSet,labels = createDataSet()
# tree = createTree(dataSet,labels)
# print(tree)
# print(dataSet)
# print(labels)
# print(calShanno(dataSet))
# print(dataSet)
# data = splitDataSet(dataSet,0,0)
# print(data)
# print(chooseBestSplit(dataSet))
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
	features = [vec[bestSplit] for vec in dataSet]
	featureSet = set(features)
	for value in featureSet:
		data = splitDataSet(dataSet,bestSplit,value)
		newLables = labels[:bestSplit] + labels[bestSplit+1:] 
		myTree[label][value] = createTree(data,newLables)
	return myTree


def classify(inputTree,featLabels,testVec):
	firstStr = list(inputTree.keys())[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	classLabel = 0
	for key in secondDict:
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key],featLabels,testVec)
			else:
				classLabel = secondDict[key]
	return classLabel
	# print(featIndext)
# a = [1,1,1,1,2,3,3,3,3,3,3]
# val = Counter(a).most_common(1)[0][0]
# print(val)









dataSet,labels = createDataSet()
tree = createTree(dataSet,labels)
label = classify(tree,labels,[1,1])
print(label)
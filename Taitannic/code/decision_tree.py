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

dataSet,labels = createDataSet()
print(dataSet)
print(labels)
print(calShanno(dataSet))
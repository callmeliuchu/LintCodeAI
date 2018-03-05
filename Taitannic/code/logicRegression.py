import math
import numpy as np
import random


def loadDataSet(path):
	dataMat = []
	labels = []
	with open(path) as f:
		for line in f.readlines():
			arr = line.strip().split()
			dataMat.append([1.0,float(arr[0]),float(arr[1])])
			labels.append(int(arr[2]))
	return dataMat,labels


def sigmoid(inX):
	return 1.0/(1+math.exp(-inX))

def gradAscent(dataMatIn,classLabels):
	dataMatrix = np.mat(dataMatIn)
	labelMat = np.mat(classLabels).transpose()
	m,n = np.shape(dataMatrix)
	alpha = 0.001
	maxCycle = 500
	weights = np.ones((n,1))
	for k in range(maxCycle):
		tmpMat = (dataMatrix*weights)
		tmpMat = np.array([[sigmoid(val)] for val in tmpMat])
		error = labelMat - tmpMat
		weights = weights + alpha*dataMatrix.transpose()*error
	return weights




def plotBestFit(weights):
	import matplotlib.pyplot as plt
	dataMat,labelMat=loadDataSet('testSet.txt')
	dataArr = np.array(dataMat)
	n = dataArr.shape[0]
	xcords1 = []
	ycords1 = []
	xcords2 = []
	ycords2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcords1.append(dataArr[i,1])
			ycords1.append(dataArr[i,2])
		else:
			xcords2.append(dataArr[i,1])
			ycords2.append(dataArr[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcords1,ycords1,s=30,c='red',marker='s')
	ax.scatter(xcords2,ycords2,s=30,c='green')
	x = np.arange(-3.0,3.0,0.1)
	y = (-weights[0]-weights[1]*x)/weights[2]
	ax.plot(x,y)
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.show()


def stocGradAscent0(dataMatrix,classLabels):
	dataMat = np.array(dataMatrix)
	m,n = np.shape(dataMat)
	alpha = 0.001
	weights = np.ones(n)
	for i in range(m):
		h = sigmoid(sum(dataMat[i]*weights))
		error = classLabels[i] - h
		weights = weights + alpha*error*dataMat[i]
	return weights


def stocRandomGradAscent0(dataMatrix,classLabels,iteratorNum=150):
	dataMat = np.array(dataMatrix)
	m,n = np.shape(dataMat)
	alpha = 0.001
	weights = np.ones(n)
	for j in range(iteratorNum):
		nums = list(range(m))
		random.shuffle(nums)
		for i in nums:
			h = sigmoid(sum(dataMat[i]*weights))
			error = classLabels[i] - h
			weights = weights + alpha*error*dataMat[i]
	return weights



dataMat,labels = loadDataSet('testSet.txt')
print(labels)
# w = gradAscent(dataMat,labels)
# w1 = stocGradAscent0(dataMat,labels)
# w2 = stocRandomGradAscent0(dataMat,labels)
# print(w)
# print(w1)
# print(w2)
# plotBestFit(w.getA())
# plotBestFit(w1)
# plotBestFit(w2)
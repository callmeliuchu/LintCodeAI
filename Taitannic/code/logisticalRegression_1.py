import numpy as np
import math
import random

def loadDataSet(fileName):
	dataMat = []
	labels= []
	with open(fileName) as f:
		for line in f.readlines():
			arr = line.strip().split()
			dataMat.append([1,float(arr[0]),float(arr[1])])
			labels.append(int(arr[2]))
	return dataMat,labels

def sigmoid(x):
	return 1/(1+math.exp(-x))

def logistic(dataMatIn,labelsIn):
	dataMat = np.mat(dataMatIn)
	labels = np.mat(labelsIn).T
	m,n = dataMat.shape
	alpha = 0.001
	cycles = 500
	w = np.ones((n,1))
	for i in range(cycles):
		h = np.array([[sigmoid(val)] for val in dataMat*w])
		error = labels - h
		w = w + alpha*dataMat.T*error
	return w

def randomLogistic(dataMatIn,labelsIn,iteratorNum=150):
	dataMat = np.mat(dataMatIn)
	m,n = np.shape(dataMat)
	w = np.ones((n,1))
	for j in range(iteratorNum):
		nums = list(range(m))
		random.shuffle(nums)
		for i in nums:
			alpha = 0.01 + 1/(1+i+j)
			error = labelsIn[i] - dataMat[i]*w
			w = w + alpha*dataMat[i].T*error
	return w

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


# dataMat,labels = loadDataSet('testSet.txt')
# w2 = randomLogistic(dataMat,labels)
# print(w2)
# w1 = logistic(dataMat,labels)
# print(w1)

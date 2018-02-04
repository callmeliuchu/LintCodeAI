import math
from collections import Counter
import numpy as np
dataSet = [[2,3],[4,7],[5,4],[7,2],[8,1],[9,6],[7,8],[5,9],[8,12]]

class Node:
	def __init__(self,split,nodeValue,leftNode,rightNode):
		self.split = split
		self.leftNode = leftNode
		self.rightNode = rightNode
		self.nodeValue = nodeValue



class Tree:
	def __init__(self,dataSet):
		self.dataSet = dataSet
	
	def root(self):
		return self.genTree(self.dataSet,0)

	def genTree(self,dataSet,index):
		data_len = len(dataSet)
		if data_len == 0:
			return None
		data = sorted(dataSet,key=lambda x:x[index])
		split = data_len//2
		nodeValue = data[split]
		aLen = len(dataSet[0])
		return Node((index+1)%aLen,nodeValue,self.genTree(data[:split],(index+1)%aLen),self.genTree(data[split+1:],(index+1)%aLen))



def visit(root):
	if root:
		print(root.nodeValue,root.split)
		visit(root.leftNode)
		visit(root.rightNode)


from math import sqrt
from collections import namedtuple


result = namedtuple("Result_tuple","nearest_point nearest_dist nodes_visited")






def find_nearest(tree,point):
	k = len(point)
	def travel(kd_node,target,max_dist):
		if kd_node is None:
			return result([0]*k,float("inf"),0)

		nodes_visited = 1
		s = kd_node.split
		pivot = kd_node.nodeValue

		if target[s] <= pivot[s]:
			nearest_node = kd_node.leftNode
			futher_node = kd_node.rightNode
		else:
			nearest_node = kd_node.rightNode
			futher_node = kd_node.leftNode

		temp1 = travel(nearest_node,target,max_dist)
		print(temp1)

		nearest = temp1.nearest_point
		dist = temp1.nearest_dist
		nodes_visited += temp1.nodes_visited

		if dist < max_dist:
			max_dist = dist

		temp_dist = abs(pivot[s] - target[s])

		if  max_dist < temp_dist:
			return result(nearest,dist,nodes_visited)

		temp_dist = math.sqrt(sum((p1-p2)**2 for p1,p2 in zip(pivot,target)))

		if temp_dist < dist:
			nearest = pivot
			dist = temp_dist
			max_dist = dist


		temp2 = travel(futher_node,target,max_dist)
		print(temp2)

		nodes_visited += temp2.nodes_visited

		if temp2.nearest_dist < dist:
			nearest = temp2.nearest_point
			dist = temp2.nearest_dist

		return result(nearest,dist,nodes_visited)

	return travel(tree,point,float("inf"))


# tree = Tree(dataSet)
# root = tree.root()
# res = find_nearest(root,[3,4.5])
# print(res)





def createDataSet():
	dataSet = [[2,3],[4,7],[5,4],[7,2]]
	labels = ['A','A','B','B']
	return dataSet,labels

dataSet,labels = createDataSet()
print(dataSet,labels)

def dis(vec1,vec2):
	return math.sqrt(sum((p1-p2)**2 for p1,p2 in zip(vec1,vec2)))


def knn_nearest(vec,dataSet,labels,k=3):
	dis_arr = [(dis(vec,dataSet[i]),labels[i]) for i in range(len(dataSet))]
	new_dis_arr = sorted(dis_arr,key=lambda x:x[0])
	count = Counter(new_dis_arr[i][1] for i in range(k))
	return count.most_common(1)[0][0]


def normalDataSet(dataSet):
	max_arr = []
	min_arr = []
	for i in range(len(dataSet[0])):
		max_val = max(vec[i] for vec in dataSet)
		min_val = min(vec[i] for vec in dataSet)
		max_arr.append(max_val)
		min_arr.append(min_val)
	max_arr = np.array(max_arr)
	min_arr = np.array(min_arr)
	range_arr = max_arr - min_arr
	range_arr[range_arr==0] = 1
	dataSet = np.array(dataSet)
	return min_arr,range_arr,(dataSet-min_arr)/range_arr

def normal_vec(min_arr,range_arr,vec):
	return (vec - min_arr)/range_arr


# res = knn_nearest([1,2],dataSet,labels)
# min_arr,range_arr,normal_data = normalDataSet(dataSet)
# print(res)








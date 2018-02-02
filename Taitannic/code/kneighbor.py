
dataSet = [[2,3],[4,7],[5,4],[7,2],[8,1],[9,6]]

class Node:
	def __init__(self,nodeValue,leftNode,rightNode):
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
		if data_len == 1:
			return Node(dataSet[0],None,None)
		data = sorted(dataSet,key=lambda x:x[index])
		split = data_len//2
		if split+1>len(data):
			return None
		nodeValue = data[split]
		return Node(nodeValue,self.genTree(data[:split],index+1),self.genTree(data[split+1:],index+1))



def visit(root):
	if root:
		print(root.nodeValue)
		visit(root.leftNode)
		visit(root.rightNode)
tree = Tree(dataSet)
root = tree.root()
visit(root)





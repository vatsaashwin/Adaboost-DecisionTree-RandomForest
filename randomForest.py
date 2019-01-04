import numpy as np
from decision_tree import DecisionTreeClassifier
from treeFunc import createTree, predictLabel
# import time

def createForest(m, n, d, data):
	# m: Number of features for a tree; n: Number of trees; d: Depth of each tree
	print("Number of features, m = {}".format(m))
	print("Number of trees, n = {}".format(n))
	print("Maximum Depth, d = {}".format(d))
	length = len(data)
	list_of_trees = []
	
	for i in range(n):
		print("Tree {}".format(i))
		list_of_trees.append(DecisionTreeClassifier())
		sample = np.random.choice(length, length, replace = True)
		sample = data[sample]
		# starttime = time.time()
		createTree(sample, d, 0, list_of_trees[i], m)
		# print("Total Training time {}".format(time.time() - starttime))
		

	return list_of_trees


def forestAccuracy(list_of_trees, data, depth):
	size = len(data)
	err = 0
	for row in range(0, size):
		n_pos = 0
		n_neg = 0
		for tree in list_of_trees:
			prediction = predictLabel(tree, data[row, :], 0, depth)
			if prediction == 1:
				n_pos = n_pos + 1
			else:
				n_neg = n_neg + 1
		if n_pos > n_neg:
			label = 1
		else:
			label = -1
		if label != data[row, 0]:
			err += 1
		accuracy = ((size - err) * 1.0/ size) * 100

	return accuracy
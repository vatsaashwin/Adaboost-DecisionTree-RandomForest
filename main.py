import numpy as np
import csv
# import time
# from matplotlib import pyplot as plt
import preprocess as prep
from treeFunc import createTree, treeAccuracy
from decision_tree import DecisionTreeClassifier
from random import sample, randint
from randomForest import createForest, forestAccuracy
from adaboost import adaboost, treeAccuracy_ada

if __name__ == '__main__':
	trainData = prep.fileRead('pa3_train_reduced.csv')	# Read Training Examples
	trainData = prep.changeData(trainData)
	validData = prep.fileRead('pa3_valid_reduced.csv')	# Read Validation Data
	validData = prep.changeData(validData)
	
	##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART-1: DECISION TREE CLASSIFIER~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
	print("!!!!!Executing DECISION TREE CLASSIFIER!!!!!")
	maximum_Depth = 20
	tree = DecisionTreeClassifier()
	# starttime = time.time()
	createTree(trainData, maximum_Depth, 0, tree, 100)
	# print("Total Training time {}".format(time.time() - starttime))

	train_acc_list = []
	valid_acc_list = []
	itr_list = []
	for i in range(21):
		itr_list.append(i)
		train_acc_list.append(treeAccuracy(tree, trainData, i))
		valid_acc_list.append(treeAccuracy(tree, validData, i))

	# plt.scatter(itr_list, train_acc_list, color = 'blue', s = 15)
	# blue_line, = plt.plot(itr_list, train_acc_list, color = 'blue', label = 'Training Accuracy')
	# plt.title("ACCURACY vs DEPTH")
	# plt.xlabel("Depth")
	# plt.ylabel("Accuracy (in %)")

	# plt.scatter(itr_list, valid_acc_list, color = 'red', s = 15)
	# red_line, = plt.plot(itr_list, valid_acc_list, color = 'red', label = 'Validation Accuracy')
	# plt.legend(handles  = [blue_line, red_line])
	# plt.grid()
	# plt.show()
	print("Training Data accuracy at each depth:")
	print(train_acc_list)
	print("Validation Data accuracy at each depth:")
	print(valid_acc_list)
	print("Print Tree Function:")
	tree.printTree()


	##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART-2: RANDOM FOREST (BAGGING)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
	
	# print("!!!!!Executing RANDOM FOREST!!!!!")
	# d = 9
	# train_acc_list = []
	# valid_acc_list = []
	# itr_list = []

	# m = 10

	# for n in [1, 2, 5, 10, 25]:
	# 	itr_list.append(n)
	# 	list_of_trees = createForest(m, n, d, trainData)
	# 	train_acc_list.append(forestAccuracy(list_of_trees, trainData, d))
	# 	valid_acc_list.append(forestAccuracy(list_of_trees, validData, d))

	# # plt.scatter(itr_list, train_acc_list, color = 'blue', s = 15)
	# # blue_line, = plt.plot(itr_list, train_acc_list, color = 'blue', label = 'Training Accuracy')
	# # plt.title("ACCURACY vs Forest Size")
	# # plt.xlabel("Forest Size")
	# # plt.ylabel("Accuracy (in %)")

	# # plt.scatter(itr_list, valid_acc_list, color = 'red', s = 15)
	# # red_line, = plt.plot(itr_list, valid_acc_list, color = 'red', label = 'Validation Accuracy')
	# # plt.legend(handles  = [blue_line, red_line])
	# # plt.grid()
	# # plt.show()
	
	# print("Training Data accuracy at each depth:")
	# print(train_acc_list)
	# print("Validation Data accuracy at each depth:")
	# print(valid_acc_list)
	# print("Print Tree Function:")


	

	##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART-3: ADABOOST (BOOSTING)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
	
	# print("!!!!!Executing ADABOOST!!!!!")
	# train_acc_list = []
	# valid_acc_list = []
	# itr_list = []
	# i = 0
	# for l in [1, 5, 10, 20]:
	# 	print("``````````````````````For {} Weak-Learners```````````````````````````".format(l))
	# 	itr_list.append(l)
	# 	tree_list, alpha_list = adaboost(trainData, l, 9)
	# 	train_acc_list.append(treeAccuracy_ada(tree_list, alpha_list, trainData, 9))
	# 	valid_acc_list.append(treeAccuracy_ada(tree_list, alpha_list, validData, 9))
	# 	i = i + 1

	# # plt.scatter(itr_list, train_acc_list, color = 'blue', s = 15)
	# # blue_line, = plt.plot(itr_list, train_acc_list, color = 'blue', label = 'Training Accuracy')
	# # plt.title("ACCURACY vs No. of Weak-Learners")
	# # plt.xlabel("No. of Weak-Learners")
	# # plt.ylabel("Accuracy (in %)")

	# # plt.scatter(itr_list, valid_acc_list, color = 'red', s = 15)
	# # red_line, = plt.plot(itr_list, valid_acc_list, color = 'red', label = 'Validation Accuracy')
	# # plt.legend(handles  = [blue_line, red_line])
	# # plt.grid()
	# # plt.show()
	# print(train_acc_list)
	# print(valid_acc_list)
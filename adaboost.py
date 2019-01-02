import numpy as np
from decision_tree import DecisionTreeClassifier
from treeFunc import predictLabel
# import time

def calcLabel(data):
	size = len(data)

	sortedValues = data[np.argsort(data[:, 0])]
	splitMat = np.split(sortedValues, np.where(sortedValues[:, 0] > 0)[0][:1])
	sumWeights = np.sum(sortedValues[:, 1])

	p_neg = 0
	p_pos = 0
	if len(splitMat[0]) >= 1 and splitMat[0][0, 0] == -1:
		p_neg = (np.sum(splitMat[0][:, 1]) * len(splitMat[0]) * 1.0)/ (sumWeights * size)
		if len(splitMat) > 1:
			p_pos =  (np.sum(splitMat[1][:, 1]) * len(splitMat[1]) * 1.0)/ (sumWeights * size)
	else:
		p_pos =  (np.sum(splitMat[1][:, 1]) * len(splitMat[1]) * 1.0)/ (sumWeights * size)

	if p_pos > p_neg:
		prediction = 1
	else:
		prediction = -1

	return prediction


def giniIndex(data):
	size = len(data)
	if size == 0:
		return 0

	sortedValues = data[np.argsort(data[:, 0])]
	splitMat = np.split(sortedValues, np.where(sortedValues[:, 0] == 1)[0][:1])

	# If only one label is present in the dataset or if the total size if only 1; return 0 because no gain achieved
	if len(splitMat[0]) == 0 or len(splitMat) == 1:
		return 0

	sumWeights = np.sum(sortedValues[:, 1])
	p_pos = (np.sum(splitMat[1][:, 1]) * len(splitMat[1]) * 1.0)/ (sumWeights * size)
	p_neg = (np.sum(splitMat[0][:, 1]) * len(splitMat[0]) * 1.0)/ (sumWeights * size)

	return (1 - p_neg ** 2 - p_pos ** 2)
    


def getInfoGain(label, data, u_root):
	# starttime = time.time()
	value = np.empty([len(data), 3])
	value[:, 0] = label[:, 0]
	value[:, 1] = label[:, 1]
	value[:, 2] = data
	sortedValues = value[np.argsort(value[:, 2])]

	# Calculate gain for every threshold in a feature
	gain = 0
	threshold = 0
	prev_label = 0
	sumWeights = np.sum(sortedValues[:, 1])

	for index in range(len(sortedValues)):
		row = sortedValues[index,:]
		thresh = row[2]
		if prev_label != row[0]:
			if index != 0:
				thresh = (sortedValues[index - 1, 2] + thresh)/ 2

			val = np.split(sortedValues, np.where(sortedValues[:, 2] >= thresh)[0][:1])
			# trueExamples = sortedValues[sortedValues[:, 1] >= thresh]
			# falseExamples = sortedValues[sortedValues[:, 1] < thresh]
			trueExamples = val[1]
			falseExamples = val[0]

			u_left = giniIndex(trueExamples[:, 0:2])
			u_right = giniIndex(falseExamples[:, 0:2])
			p_left = (np.sum(trueExamples[:, 1]) * len(trueExamples) * 1.0)/ (sumWeights * len(sortedValues))
			p_right = (np.sum(falseExamples[:, 1]) * len(falseExamples) * 1.0)/ (sumWeights * len(sortedValues))

			currentGain = u_root - p_left * u_left - p_right * u_right


			if currentGain > gain:
				gain = currentGain
				threshold = thresh
			prev_label = row[0]
	
	# print("Gain = {} | Threshold = {} | Time = {}".format(gain, threshold, time.time() - starttime))
	return gain, threshold


def createTree_adaboost(data, maximumDepth, currentDepth, tree):
	print("At depth: {}".format(currentDepth))
	# starttime = time.time()

	if currentDepth == maximumDepth:
		label = calcLabel(data[:, 0:2])
		tree.insert(None, None, True, label)
		return

	u_root = giniIndex(data[:, 0:2])
	
	gain = 0
	threshold = 0
	best_feature = 0
	for featureIndex in range(2, data.shape[1]):
		print("Calculating Gain for Feature Number: {}".format(featureIndex - 2))
		currentGain, currentThreshold = getInfoGain(data[:, 0:2], data[:, featureIndex], u_root)
		if currentGain > gain:
			gain = currentGain
			threshold = currentThreshold
			best_feature = featureIndex

	if gain == 0:
		label = calcLabel(data[:, 0:2])
		tree.insert(None, None, True, label)
		return
	
	trueExamples = data[data[:,best_feature] >= threshold]
	falseExamples = data[data[:,best_feature] < threshold]

	label = calcLabel(data[:, 0:2])
	tree.insert(best_feature, threshold, False, label)
	tree.left = DecisionTreeClassifier()
	tree.right = DecisionTreeClassifier()
	currentDepth = currentDepth + 1
	# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Time for depth: {} = {}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~".format(currentDepth, time.time() - starttime))
	createTree_adaboost(trueExamples, maximumDepth, currentDepth, tree.left)
	createTree_adaboost(falseExamples, maximumDepth, currentDepth, tree.right)


def errorCalc(tree, data, maxDepth):
	error = 0
	signed_product_y_hypo = []
	for row in data:
		prediction = predictLabel(tree, row, 0, maxDepth)

		if row[0] != prediction:
			error = error + row[1]
			hypothesis_times_y = 1
		else:
			hypothesis_times_y = -1
		signed_product_y_hypo.append(hypothesis_times_y)
	error = error/ np.sum(data[:, 1])
	return error, signed_product_y_hypo


def adaboost(data, l, maximumDepth):
	size = len(data)
	D = np.empty(size)	# D is the Distribution Matrix (Matrix containing the weights)
	D.fill(1.0/ size)

	data = np.insert(data, 1, D, axis = 1)	# Insert D as column indexed at 1 in data
	
	tree_list = []
	alpha_list = []

	for weakLearner in range(l):
		tree = DecisionTreeClassifier()
		print("Learner No: {}".format(weakLearner))
		createTree_adaboost(data, maximumDepth, 0, tree)
		err, weightChange_list = errorCalc(tree, data, maximumDepth)    
		alpha = (np.log(((1 - err) * 1.0)/ err))/ 2
		data[:, 1] = data[:, 1] * np.exp(alpha * np.array(weightChange_list))
		tree_list.append(tree)
		alpha_list.append(alpha)

	return tree_list, alpha_list


def treeAccuracy_ada(tree_list, alpha_list, data, maxDepth):
	size = len(data)
	error = 0
	D = np.empty(size) 
	data = np.insert(data, 1, D, axis = 1)	

	for row in data:
		sumWeights = 0
		index = 0
		for tree in tree_list:
			alpha = alpha_list[index]
			prediction = predictLabel(tree, row, 0, maxDepth)
			sumWeights = sumWeights + prediction * alpha
			index = index + 1
		if np.sign(sumWeights) != row[0]:
			error = error + 1

	accuracy = ((size - error) * 1.0 / size) * 100
	return accuracy

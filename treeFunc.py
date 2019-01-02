import numpy as np
# import time
from decision_tree import DecisionTreeClassifier
from random import sample


def calcLabel(data):
	length = len(data)
	
	unique, count = np.unique(data, return_counts = True)	# Get a list containing count of positive and negative labels
	if len(unique) == 1:
		# If only one label is present, return that label
		if unique[0] == 1:
			return 1
		else:
			return -1
	if unique[0] == 1:
		p_pos = (count[0] * 1.0)/ length
		p_neg = (count[1] * 1.0)/ length
	else:
		p_pos = (count[1] * 1.0)/ length
		p_neg = (count[0] * 1.0)/ length
	if p_pos >= p_neg:
		return 1
	else:
		return -1


def giniIndex(data):
	n = len(data)
	if n == 0:
		return 0
	unique, count = np.unique(data, return_counts = True)
	if len(unique) == 1:
		return 0
	if unique[0] == 1:
		p_pos = (count[0] * 1.0)/ n
		p_neg = (count[1] * 1.0)/ n
	else:
		p_pos = (count[1] * 1.0)/ n
		p_neg = (count[0] * 1.0)/ n
 
	return (1 - (p_pos ** 2) - (p_neg ** 2))

def getInfoGain(label, data, u_root):
	# starttime = time.time()
	value = np.empty([len(data), 2])
	value[:, 0] = label
	value[:, 1] = data
	sortedValues = value[np.argsort(value[:, 1])]

	# Calculate gain for every threshold in a feature
	gain = 0
	threshold = 0
	prev_label = 0

	for index in range(len(sortedValues)):
		row = sortedValues[index,:]
		thresh = row[1]
		if prev_label != row[0]:
			if index != 0:
				thresh = (sortedValues[index - 1, 1] + thresh)/ 2

			val = np.split(sortedValues, np.where(sortedValues[:, 1] >= thresh)[0][:1])
			# trueExamples = sortedValues[sortedValues[:, 1] >= thresh]
			# falseExamples = sortedValues[sortedValues[:, 1] < thresh]
			trueExamples = val[1]
			falseExamples = val[0]

			u_left = giniIndex(trueExamples[:, 0])
			u_right = giniIndex(falseExamples[:, 0])
			p_left = len(trueExamples) * 1.0/ len(sortedValues)
			p_right = len(falseExamples) * 1.0/ len(sortedValues)

			currentGain = u_root - p_left * u_left - p_right * u_right


			if currentGain > gain:
				gain = currentGain
				threshold = thresh
			prev_label = row[0]
	
	# print("Gain = {} | Threshold = {} | Time = {}".format(gain, threshold, time.time() - starttime))
	return gain, threshold

def createTree(data, maximumDepth, currentDepth, tree, m):
	print("At depth: {}".format(currentDepth))
	# starttime = time.time()

	if currentDepth == maximumDepth:
		# If you have reached maximum depth, store the prediction based on number of positive and negative examples
		label = calcLabel(data[:, 0])
		tree.insert(None, None, True, label)
		return

	u_root = giniIndex(data[:, 0])
	
	gain = 0
	threshold = 0
	best_feature = 0
	i = 0
	for featureIndex in sample(range(1, data.shape[1]), m):
		print("Calculating Gain for Feature Number:{}".format(i))
		currentGain, currentThreshold = getInfoGain(data[:, 0], data[:, featureIndex], u_root)
		if currentGain > gain:
			gain = currentGain
			threshold = currentThreshold
			best_feature = featureIndex
		i = i + 1

	if gain == 0:
		label = calcLabel(data[:, 0])
		tree.insert(None, None, True, label)
		return
	
	trueExamples = data[data[:,best_feature] >= threshold]
	falseExamples = data[data[:,best_feature] < threshold]

	label = calcLabel(data[:, 0])
	tree.insert(best_feature, threshold, False, label)
	tree.left = DecisionTreeClassifier()
	tree.right = DecisionTreeClassifier()
	currentDepth = currentDepth + 1
	# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Time for depth: {} = {}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~".format(currentDepth, time.time() - starttime))
	createTree(trueExamples, maximumDepth, currentDepth, tree.left, m)
	createTree(falseExamples, maximumDepth, currentDepth, tree.right, m)


def predictLabel(tree, example, currentDepth, maximumDepth):
	if tree.is_leaf:
		return tree.prediction
	if currentDepth == maximumDepth:
		return tree.prediction

	if example[tree.splitFeature].reshape(1,1) >= tree.threshold:
		return predictLabel(tree.left, example, currentDepth + 1, maximumDepth)
	else:
		return predictLabel(tree.right, example, currentDepth + 1, maximumDepth)

def treeAccuracy(tree, data, maximumDepth):
	size = len(data)
	err = 0
	for row in range(0, size):
		prediction = predictLabel(tree, data[row, :], 0, maximumDepth)
		if prediction != data[row, 0]:
			err = err + 1
	accuracy = ((size - err) * 1.0/ size) * 100
	return accuracy
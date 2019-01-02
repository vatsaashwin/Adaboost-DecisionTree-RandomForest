# Define a Class for Decision tree, each decision tree node will be an object of class- DecisionTreeClassifier
class DecisionTreeClassifier:

	def __init__(self):
		# Initializer function for class
		self.leftNode = None
		self.rightNode = None
		self.splitFeature = None
		self.threshold = None
		self.prediction = None
		self.is_leaf = None

	def insert(self, splitFeature, threshold, is_leaf, prediction):
		# This function inserts values in the leaf node
		self.splitFeature = splitFeature
		self.threshold = threshold
		self.is_leaf = is_leaf
		self.prediction = prediction

	def printTree(self, spacing=""):
		# This function prints the tree
		# Base case: we've reached a leaf
		if self.is_leaf:
			print(spacing + "Predict", self.prediction)
			return
		print("Feature : ", self.splitFeature, " Threshold : ", self.threshold)
		# Call this function recursively on the true branch
		print(spacing + '--> True:')
		self.left.printTree()
		# Call this function recursively on the false branch
		print(spacing + '--> False:')
		self.right.printTree()
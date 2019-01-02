import numpy as np
import csv

def fileRead(fileName):
	data = np.genfromtxt(fileName, delimiter = ',')
	return data

def changeData(data):
	'''CHANGE 3 TO 1 AND 5 TO -1'''
	data[:, 0] = np.where(data[:, 0] > 3.0, -1, 1)
	return data
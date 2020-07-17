# Based on following along with Single-Layer Perceptron: Background & Python Code -- Brian Faure
# https://youtu.be/OVHc-7GYRo4
# Notes added for later review.

from __future__ import print_function
import sys
from matplotlib import pyplot as plt 
import numpy as np 

def plot(matrix, weights = None, title = "Prediction Matrix"):
	#if 1D inputs, excluding bias and ys
	if len(matrix[0]) == 3:
		fig, ax = plt.subplots()
		ax.set_title(title)	#title of the table
		ax.set_xlabel("i1")	#setting the x axis labels
		ax.set_ylabel("Classifications")	#Setting the y axis label

		if weights != None:
			# Range of view shown by output table
			y_min = -0.1
			y_max = 1.1
			x_min = 0.0
			x_max = 1.1
			y_res = 0.001
			x_res = 0.001

			#Areas for points on graph
			ys = np.arange(y_min, y_max, y_res)
			xs = np.arange(x_min, x_max, x_res)
			zs = []

			for cur_y in np.arange(y_min, y_max, y_res):
				for cur_x in np.arange(x_min, x_max, x_res):
					zs.append(predict([1.0, cur_x], weights))

			xs, ys = np.meshgrid(xs, ys)
			zs = np.array(zs)
			zs = zs.reshape(xs.shape)
			cp = plt.contourf(xs, ys, zs, levels = [-1, -0.001, 0, 1], colors = ('b', 'r'), alpha = 0.1)

		c1_data = [[], []]
		c0_data = [[], []]

		for i in range(len(matrix)):
			cur_i1 = matrix[i][j]
			cur_y = matrix[i][-1]

			if cur_y == 1:
				c1_data[0].append(cur_i1)
				c1_data[1].append(1.0)
			else:
				c0_data[0].append(cur_i1)
				c0_data[1].append(0.0)

		plt.xticks(np.arange(x_min, x_max, 0.1))	#Steps from min to max on x axis move by .1 space.
		plt.yticks(np.arange(y_min, y_max, 0.1))	#Steps from min to max on y axis move by .1 space.
		plt.xlim(0, 1.05)
		plt.ylim(-0.05, 1.05)

		c0s = plt.scatter(c0_data[0], c0_data[1], s = 40.0, c = 'r', label = 'Class -1')
		c1s = plt.scatter(c1_data[0], c1_data[1], s = 40.0, c = 'b', label = 'Class 1')

		plt.legend(fontsize = 10, loc = 1)
		plt.show()
		return

	# if 2D inputs, exclude bias and ys.
	# A different type of plot based on matrix size.
	if len(matrix[0]) == 4:
		fig, ax = plt.subplots()
		ax.set_title(title)	#title of the table
		ax.set_xlabel("i1")	#setting the x axis labels
		ax.set_ylabel("i2")	#Setting the y axis label

		if weights != None:
			# Range of view shown by output table
			map_min = 0.0
			map_max = 1.1
			y_res = 0.001
			x_res = 0.001

			#Areas for points on graph
			ys = np.arange(map_min, map_max, y_res)
			xs = np.arange(map_min, map_max, x_res)
			zs = []

			for cur_y in np.arange(map_min, map_max, y_res):
				for cur_x in np.arange(map_min, map_max, x_res):
					zs.append(predict([1.0, cur_x, cur_y], weights))

			xs, ys = np.meshgrid(xs, ys)
			zs = np.array(zs)
			zs = zs.reshape(xs.shape)
			cp = plt.contourf(xs, ys, zs, levels = [-1, -0.001, 0, 1], colors = ('b', 'r'), alpha = 0.1)

		c1_data = [[], []]
		c0_data = [[], []]

		for i in range(len(matrix)):
			cur_i1 = matrix[i][1]
			cur_i2 = matrix[i][2]
			cur_y = matrix[i][-1]

			if cur_y == 1:
				c1_data[0].append(cur_i1)
				c1_data[1].append(cur_i2)
			else:
				c0_data[0].append(cur_i1)
				c0_data[1].append(cur_i2)

		plt.xticks(np.arange(0.0, 1.1, 0.1))	#Steps from min to max on x axis move by .1 space.
		plt.yticks(np.arange(0.0, 1.1, 0.1))	#Steps from min to max on y axis move by .1 space.
		plt.xlim(0, 1.05)
		plt.ylim(0, 1.05)

		c0s = plt.scatter(c0_data[0], c0_data[1], s = 40.0, c = 'r', label = 'Class -1')
		c1s = plt.scatter(c1_data[0], c1_data[1], s = 40.0, c = 'b', label = 'Class 1')

		plt.legend(fontsize = 10, loc = 1)
		plt.show()
		return

	print("Matrix dimensions not covered.")

def predict(inputs, weights):
	threshold = 0.0
	total_activation = 0.0
	for input, weight in zip(inputs, weights):
		total_activation += input * weight
	return 1.0 if total_activation >= threshold else 0.0

# Function for calculating the accuracy of a prediction based
# on the provided inputs and associated weights.
# each matrix row: up to the last row = inputs, last row = y (classification)
def accuracy(matrix, weights):
	num_correct = 0.0
	preds = []
	for i in range(len(matrix)):
		#Get predicted classification
		pred = predict(matrix[i][:-1], weights)
		preds.append(pred)

		#check if prediction is accurate
		if pred == matrix[i][-1]: num_correct += 1.0

	print("Predictions:", preds)

	#return overall prediction accuracy
	return num_correct / float(len(matrix))

#funtion used to train the perceptron on data in the matrix.
#Training weights are returned at the end.
def train_weights(matrix, weights, nb_epoch = 10, l_rate = 1.0, do_plot = False, stop_early = True, verbose = True):
	# nb_epoch == how many times it will try to go through and train the weights
	# learn rate (l_rate) == set to a default; at lower rates the weight would be changed by a smaller amount each round.  Increase, changes weight more.  May slowdown training.
	# do_plot == If set to true, each epoch would plot the outcome
	# stop_early == If accuracy of 100 found before max epoch, can quit early
	# verbose == printing out more information to the terminal.

	#iterate for the number of epochs requested
	for epoch in range(nb_epoch):
		#calculate the current accuracy
		cur_acc = accuracy(matrix, weights)
		#print info
		print("\n Epoch %d \nWeights: " %epoch, weights)
		print("Accuracy: ", cur_acc)

		#check if training is done.
		if cur_acc == 1.0 and stop_early: break

		#check if we should plot current results
		if do_plot: plot(matrix, weights, title = "Epoch %d" %epoch)

		#iterate over each training input otherwise
		for i in range(len(matrix)):
			#calculate predictions
			prediction = predict(matrix[i][:-1], weights)
			#calculate error
			error = matrix[i][-1] - prediction

			if verbose:
				print("Training on data at index %d ..." %i)

			#iterate over reach weight and update it
			for j in range(len(weights)):
				if verbose: sys.stdout.write("\tWeight[%d]: %0.5f --> " %(j, weights[j]))
				weights[j] = weights[j] + (l_rate * error * matrix[i][j])
				if verbose: sys.stdout.write("%0.5f\n" %weights[j])
	
	#plot out the final results
	plot(matrix, weights, title = "Final Epoch")
	return weights

def main():

	#			Bias	x1	x2		y
	data = [	[1.00, 0.08, 0.72, 1.0],
				[1.00, 0.10, 1.00, 0.0],
				[1.00, 0.26, 0.58, 1.0],
				[1.00, 0.35, 0.95, 0.0],
				[1.00, 0.45, 0.15, 1.0],
				[1.00, 0.60, 0.30, 1.0],
				[1.00, 0.70, 0.65, 0.0],
				[1.00, 0.92, 0.45, 0.0]]

	weights = [0.20, 1.00, -1.00]
	train_weights(data, weights = weights, nb_epoch = 10, l_rate = 1.0, do_plot = True, stop_early = True)
	#Pop up will appear for each round at this point.  Close it for the process to continue.

if __name__ == '__main__':
	main()

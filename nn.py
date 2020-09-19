import numpy as np

class NeuralNet:
	def __init__(self, inodes, hnodes, onodes, lr):
		self.inodes = inodes
		self.hnodes = hnodes
		self.onodes = onodes
		self.lr = lr

		self.wih = np.random.random((hnodes, inodes))
		self.who = np.random.random((onodes, hnodes))

		pass

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def query(self, inputs):
		inputs = np.array(inputs, ndmin=2).T

		hidden_outputs = self.sigmoid(self.wih.dot(inputs))
		final_outputs = self.sigmoid(self.who.dot(hidden_outputs))

		return final_outputs

	def train(self, inputs, targets):
		inputs = np.array(inputs, ndmin=2).T
		targets = np.array(targets, ndmin=2).T
		
		hidden_outputs = self.sigmoid(self.wih.dot(inputs))
		final_outputs = self.sigmoid(self.who.dot(hidden_outputs))

		final_errors = targets - final_outputs
		hidden_errors = self.who.T.dot(final_errors)

		self.who += self.lr * np.dot((final_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs.T)
		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs
			)), inputs.T)
		pass

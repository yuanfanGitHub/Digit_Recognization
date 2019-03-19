from mnist.loader import MNIST
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from scipy.stats import truncnorm

def check_same_list(x,y):
	assert x.size == y.size
	for i in range(1, x.size):
		if (x[i] != y[i]):
			return False
	return True

def sigmoid(Z):
	A = 1 / (1 + np.exp(-Z))
	return A, Z
def sigmoid_gradient(dA, Z):
	A, Z = sigmoid(Z)
	dZ = dA * A * (1 - A)
	return dZ


def softmax(x):
	"""Compute the softmax of vector x in a numerically stable way."""
	shiftx = x - np.max(x)
	exps = np.exp(shiftx)
	return (exps / np.sum(exps)),x

count=0
def softmax_grad(AL, y, A_prev):
	grad, AL = softmax(AL)
	i = count%10
	for j in range(grad.shape(0)):
		if (i == j):
			gradient[j] = grad[j]*(1 - grad[i])*A_prev
		else:
			gradient[j] = grad[j]*(0 - grad[i])*A_prev
	count +=1
	return gradient

def ReLU(x):
	return x * (x > 0)

def convert_digit_to_array(x):
	a = np.array([0]*10)
	a[x] = 1
	a = np.array(a, ndmin = 1)
	return a

def exp_normalize(x):
	b = x.max()
	y = np.exp(x - b)
	return y / y.sum()

def truncated_normal(mean=0, sd=1, low=0, upp=10):
	return truncnorm(
		(low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def judge(x):
	a = np.array(x)
	a[np.where(a==np.max(a))] = 1
	a[np.where(a!=np.max(a))] = 0
	return a


def locate_max(x):
	return np.argmax(x)

class NN:
	def __init__(self, layers_dims, learning_rate, num_iterations,
			print_cost, hidden_layers_activation_fn):
		# self.images, self.labels = images, labels
		self.layers_dims = layers_dims
		self.learning_rate = learning_rate
		self.num_iterations = num_iterations
		self.print_cost = print_cost
		self.hidden_layers_activation_fn = hidden_layers_activation_fn
		self.parameters = self.initialize_parameters(layers_dims)


	def initialize_parameters(self,layers_dims):
		np.random.seed(1)	
		layers_dims = self.layers_dims		   
		parameters = {}

		L = len(layers_dims)			
		for l in range(1, L):		   
			parameters["W" + str(l)] = np.random.randn(
				layers_dims[l], layers_dims[l - 1]) * 0.01
			parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

			assert parameters["W" + str(l)].shape == (
				layers_dims[l], layers_dims[l - 1])
			assert parameters["b" + str(l)].shape == (layers_dims[l], 1)
		return parameters

	def linear_forward(self, A_prev, W, b):
		Z = np.dot(W, A_prev) + b
		cache = (A_prev, W, b)
		return Z, cache


	def linear_activation_forward(self,A_prev, W, b, activation_fn):
		assert activation_fn == "sigmoid" or activation_fn == "softmax" or \
			activation_fn == "relu" 

		if activation_fn == "sigmoid":
			Z, linear_cache = self.linear_forward(A_prev, W, b)
			A, activation_cache = sigmoid(Z)

		elif activation_fn == "softmax":
			Z, linear_cache = self.linear_forward(A_prev, W, b)
			A, activation_cache = softmax(Z)

		elif activation_fn == "relu":
			Z, linear_cache = self.linear_forward(A_prev, W, b)
			A, activation_cache = relu(Z)

		assert A.shape == (W.shape[0], A_prev.shape[1])

		cache = (linear_cache, activation_cache)
		return A, cache


	def L_model_forward(self, X):
		parameters = self.parameters
		hidden_layers_activation_fn = self.hidden_layers_activation_fn
		A = X						   
		caches = []					 
		L = len(parameters) // 2		

		for l in range(1, L):
			A_prev = A
			A, cache = self.linear_activation_forward(
				A_prev, parameters["W" + str(l)], parameters["b" + str(l)],
				activation_fn=hidden_layers_activation_fn)
			caches.append(cache)

		AL, cache = self.linear_activation_forward(
			A, parameters["W" + str(L)], parameters["b" + str(L)],
			activation_fn="sigmoid")
		caches.append(cache)

		assert AL.shape == (10,1)
		return AL, caches


	# Compute cross-entropy cost
	def compute_cost(self,AL, y):

		m = y.shape[0]			  
		cost = - (1 / m) * np.sum(
			np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL)))
		return cost


	# define helper functions that will be used in L-model back-prop
	def linear_backward(self, dZ, cache):
		A_prev, W, b = cache
		m = A_prev.shape[0]

		dW = (1 / m) * np.dot(dZ, A_prev.T)
		db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
		dA_prev = np.dot(W.T, dZ)

		assert dA_prev.shape == A_prev.shape
		assert dW.shape == W.shape
		assert db.shape == b.shape

		return dA_prev, dW, db


	def linear_activation_backward(self, dA, A, y, cache, activation_fn):
		# A_prev, W, b = cache
		linear_cache, activation_cache = cache

		if activation_fn == "sigmoid":
			dZ = sigmoid_gradient(dA, activation_cache)
			dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

		elif activation_fn == "softmax":
			dZ = softmax_grad(A, y, A_prev)
			dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
		return dA_prev, dW, db


	def L_model_backward(self,AL, y, caches, hidden_layers_activation_fn="sigmoid"):
		y = y.reshape(AL.shape)
		L = len(caches)
		grads = {}
		A = AL
		y = y
		dAL = np.divide(AL - y, np.multiply(AL, 1 - AL))#10*1
		grads["dA" + str(L - 1)], grads["dW" + str(L)], grads[
			"db" + str(L)] = self.linear_activation_backward(
				dAL, A, y, caches[L - 1], "sigmoid")

		for l in range(L - 1, 0, -1):
			current_cache = caches[l - 1]
			grads["dA" + str(l - 1)], grads["dW" + str(l)], grads[
				"db" + str(l)] = self.linear_activation_backward(
					grads["dA" + str(l)], A, y, current_cache,
					hidden_layers_activation_fn)
		return grads

	def update_parameters(self, parameters, grads, learning_rate):
		L = len(parameters) // 2

		for l in range(1, L + 1):
			parameters["W" + str(l)] = parameters[
				"W" + str(l)] - learning_rate * grads["dW" + str(l)]
			parameters["b" + str(l)] = parameters[
				"b" + str(l)] - learning_rate * grads["db" + str(l)]
		return parameters



	def L_layer_model(self, X, y):
		np.random.seed(1)

		# initialize parameters
		parameters = self.initialize_parameters(self.layers_dims)

		# intialize cost list
		cost_list = []

		# iterate over num_iterations
		for i in range(self.num_iterations):
			# iterate over L-layers to get the final output and the cache
			AL, caches = self.L_model_forward(
				X)

			# compute cost to plot it

			cost = self.compute_cost(AL, y)

			# iterate over L-layers backward to get gradients

			grads = self.L_model_backward(AL, y, caches, self.hidden_layers_activation_fn)

			# update parameters
			parameters = self.update_parameters(parameters, grads, self.learning_rate)

			# append each 100th cost to the cost list
			# if (i + 1) % 100 == 0 and print_cost:
			# 	print(f"The cost after {i + 1} iterations is: {cost:.4f}")

			# if i % 10000 == 0:
			# 	cost_list.append(cost)

		# # plot the cost curve
		# plt.figure(figsize=(10, 6))
		# plt.plot(cost_list)
		# plt.xlabel("Iterations (per hundreds)")
		# plt.ylabel("Loss")
		# plt.title(f"Loss curve for the learning rate = {self.learning_rate}")

		return parameters

	def accuracy(self, X, y):
		probs, caches = self.L_model_forward(X)
		# labels = (probs >= 0.5) * 1
		labels = judge(probs)
		# assert labels.size == y.size
		print(check_same_list(labels,y))
		unaccuracy = np.mean(labels != y) * 100
		# return f"The unaccuracy rate is: {unaccuracy:.2f}%."
		return unaccuracy


	def forward(self, input_vector, label):
		AL = L_model_forward(input_vector, self.parameters, 'sigmoid')
		target = convert_label_to_array(label)
		# compute loss
		loss = compute_cost(AL, target)
		return loss

	def backward(self, input_vector, label):
		#update gradient
		target = convert_label_to_array(label)
		grads = L_model_backward(AL, target, caches, hidden_layers_activation_fn="sigmoid")
		return grads

		



if __name__ == '__main__':

	mndata = MNIST('./Digitsets')
	mndata.gz = True
	(images, labels) = mndata.load_training()
	print(mndata.display(images[0]))
	print(labels[0])
	images = np.array(images)
	labels = np.array(labels)
	model1 = NN([784, 5, 10], 0.1, 1, True, "sigmoid")
	c = (np.array(images[0])[np.newaxis,:])/255
	d = convert_digit_to_array(labels[0])

	i = 0 
	for image,label in zip(images,labels):#60000
		a = (np.array(images[i])[np.newaxis,:])/255
		b = convert_digit_to_array(labels[i])
		model1.L_layer_model(a.T, b)
		i += 1
	i = 0
	for image,label in zip(images,labels):
		c = (np.array(images[i])[np.newaxis,:])/255
		d = convert_digit_to_array(labels[i])
		model1.accuracy(c.T,d)
		i += 1

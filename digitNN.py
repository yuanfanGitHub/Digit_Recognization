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



# def activation_function(x):
# 	return softmax(x)

def sigmoid(Z):
	A = 1 / (1 + np.exp(-Z))
	return A, Z
def sigmoid_gradient(dA, Z):
	A, Z = sigmoid(Z)
	dZ = dA * A * (1 - A)
	# print('dz')
	# print(dZ.shape)
	return dZ


def softmax(x):
	"""Compute the softmax of vector x in a numerically stable way."""
	shiftx = x - np.max(x)
	exps = np.exp(shiftx)
	return (exps / np.sum(exps)),x

count=0
def softmax_grad(AL, y, A_prev):
	#Wi
	# Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
	# s = softmax.reshape(-1,1)
	# return np.diagflat(s) - np.dot(s, s.T)
	# A,Z = softmax(Z)
	# # dZ = dA*(-dA)*Z
	# # return dZ
	# dx = A * dA
	# s = dx.sum(axis=dx.ndim - 1, keepdims=True)
	# dx -= A * s
	# if (i == j):
	# 	pass

	# else:
	# 	pass
	# print(AL.shape)
	# m = y.shape[0]
	grad, AL = softmax(AL)
	i = count%10
	for j in range(grad.shape(0)):
		if (i == j):
			gradient[j] = grad[j]*(1 - grad[i])*A_prev
		else:
			gradient[j] = grad[j]*(0 - grad[i])*A_prev
	# # print(m)
	# # print(range(m))
	# grad[range(m),y.argmax(axis=1)] -= 1
	# grad = grad/m
	# sAl,x = softmax(AL)
	# grad = np.zeros(10,5)
	# print(AL.shape)
	# m = y.shape[0]
	# grad, AL = softmax(AL)
	# print(grad.shape)
	# grad[locate_max(y),1] -= 1
	# grad = grad/m
	count +=1
	return gradient












	# return dx

def ReLU(x):
	return x * (x > 0)


def convert_digit_to_array(x):
	a = np.array([0]*10)
	a[x] = 1
	a = np.array(a, ndmin = 1)
	return a



# def convert_label_to_array(labels):
# 	return (np.array(convert_digit_to_array(labels)))[np.newaxis,:]

# using matplotlib
# imgplot = plt.imshow(images)


def exp_normalize(x):
	b = x.max()
	y = np.exp(x - b)
	return y / y.sum()

def truncated_normal(mean=0, sd=1, low=0, upp=10):
	return truncnorm(
		(low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def judge(x):
	a = np.array(x)
	# b = np.zeros_like(x)
	# b[np.arange(len(x)), a.argmax(1)] = 1
	a[np.where(a==np.max(a))] = 1
	a[np.where(a!=np.max(a))] = 0
	return a
	# return x


def locate_max(x):
	return np.argmax(x)


# from mnist import MNIST
#import mnist
#Image Module

# x = [0, 1, 2, 1]
# y = [1, 2, 1, 0]

# fig, ax = plt.subplots()
# ax.fill(x, y)
# plt.show()

# mndata = MNIST('./Digitsets')#fold where sets are

# images, labels = mndata.load_training()
# index = random.randrange(0, len(images))  # choose an index ;-)
# print(mndata.display(images[index]))
# from mnist import MNIST
#mndata = MNIST('D:\BachelorThesis\DigitRecog\Digitsets')




# index = random.randrange(0, len(images))  # choose an index ;-)
# # index = 2
# print(mndata.display(images[0]))
# # print(labels[0])
# # print(labels)
# #image size 28*28
# # print(np.ndarray.size(images[index]))

# # input1 = np.array(images[0])
# input1 = np.array(images[0],ndmin=2)
# input_size = len(images[index])
# print(input_size)
# # input1_t = np.matrix.transpose(input1)
# # print(np.multiply(input1_t,))

# # np.resize(input1, (1, input_size))

# target_vector = np.array([0]*10)
# target_vector[labels[0]] = 1
# target_vector = np.array(target_vector, ndmin = 1)
# print(target_vector)


# no_of_hidden_nodes = 5
# no_of_output = 10

# # rad = 1 / np.sqrt(no_of_hidden_nodes)
# wih = np.random.rand(input_size, no_of_hidden_nodes)
# # print(wih)

# # weights matrix from inputs to hidden nodes


# who = np.random.rand(no_of_hidden_nodes, no_of_output)
# # weights matrix from hidden nodes t output
# # print(np.dot(np.dot(input1, wih),who)) #check size






# class NeuralNetwork:
# 	def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes, learning_rate):
# 		self.no_of_in_nodes = no_of_in_nodes
# 		self.no_of_out_nodes = no_of_out_nodes
# 		self.no_of_hidden_nodes = no_of_hidden_nodes
# 		self.learning_rate = learning_rate
# 		self.creat_weights_matrix()

# 	def creat_weights_matrix(self):
# 		# rad = 1 / np.sqrt(self.no_of_in_nodes)
# 		# X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
# 		# self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, 
# 		#								self.no_of_in_nodes))
# 		# rad = 1 / np.sqrt(self.no_of_hidden_nodes)
# 		# X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
# 		# self.weights_hidden_out = X.rvs((self.no_of_out_nodes, 
# 		#								 self.no_of_hidden_nodes))
# 		self.weights_in_hidden = np.random.normal(0,1,size=(self.no_of_in_nodes, self.no_of_hidden_nodes))
# 		self.weights_hidden_out = np.random.normal(0,1,size=(self.no_of_hidden_nodes, self.no_of_out_nodes))

# 	def run(self, input_vector):
# 		input_vector1 = (np.array(input_vector))[np.newaxis,:] /256
# 		# print(input_vector1.shape)  #(1,784)
# 		output_vector2 = np.dot(input_vector1, self.weights_in_hidden)
# 		# print(output_vector2.shape)  #(1,5)
# 		output_vector3 = sigmoid(output_vector2)
# 		output_vector4 = np.dot(output_vector3, self.weights_hidden_out)
# 		# print(output_vector4.shape) #(1,10)
# 		output_vector5 = activation_function(output_vector4)
# 		# print(judge(output_vector4))
# 		return output_vector5

# 	def train(self, input_vector, label):
# 		input_vector1 = (np.array(input_vector)[np.newaxis,:])/256#(1,784)/256
# 		# target_vector = np.array(target_vector, ndmin = 1)
# 		# convert_digit_to_array
# 		# print(label)
# 		target_vector1 = (np.array(convert_digit_to_array(label)))[np.newaxis,:]#(1,10)
# 		# target_vector = activation_function(target_vector)

# 		output_vector1 = np.dot(input_vector1, self.weights_in_hidden)
# 		output_hidden = sigmoid(output_vector1)
# 		output_vector2 = np.dot(output_hidden, self.weights_hidden_out)
# 		output_vector3 = output_vector2
# 		# print(output_vector3)
# 		output_network = activation_function(output_vector3)
# 		# print(output_network)
# 		# output_network = output_vector2
# 		# print(output_network)
# 		# output_network_normalized = exp_normalize(output_network)
# 		# output_network = output_network_normalized
# 		# output_errors =  target_vector - output_network
# 		output_errors =   -output_network + target_vector1
# 		# errors_out = (target_vector - output_network) * (target_vector - output_network) * 0.5
# 		#squared error
# 		#update weights 
# 		# tmp = np.multiply(output_errors , np.multiply(output_network , (1.0 - output_network)))
# 		tmp = output_errors * output_network * (1.0 - output_network)
# 		# tmp = tmp[np.newaxis,:]

# 		# output_hidden = output_hidden[:,np.newaxis]
# 		# np.resize(output_hidden,(5,1))
# 		# print(output_hidden.shape)
# 		# np.resize(tmp,(1,10))
# 		# tmp1 = self.learning_rate  * np.dot(output_hidden[:,np.newaxis],tmp)
# 		tmp1 = self.learning_rate  * np.dot(output_hidden.T, tmp)
# 		# print(np.size(tmp1))
# 		self.weights_hidden_out -= tmp1
# 		# print(self.weights_hidden_out)


# 		hidden_errors = np.dot(output_errors, self.weights_hidden_out.T)
# 		# print(np.size(hidden_errors))
# 		# tmp2 = np.multiply(hidden_errors, np.multiply(output_hidden , (1.0 - output_hidden)))
# 		tmp2 = hidden_errors * output_hidden * (1.0 - output_hidden)
# 		# print(np.size(1.0 - output_hidden))
# 		# print(np.size(tmp2)) #100
# 		# self.weights_in_hidden -= self.learning_rate * np.dot((input_vector[np.newaxis,:]).T, tmp2[np.newaxis,:])
# 		self.weights_in_hidden -= self.learning_rate * np.dot((input_vector1).T, tmp2)
# 		# print(np.size(np.multiply(input_vector.T,tmp2)))




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
		# print(layers_dims)
		# assert False
		L = len(layers_dims)			
		for l in range(1, L):		   
			parameters["W" + str(l)] = np.random.randn(
				layers_dims[l], layers_dims[l - 1]) * 0.01
			parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

			assert parameters["W" + str(l)].shape == (
				layers_dims[l], layers_dims[l - 1])
			assert parameters["b" + str(l)].shape == (layers_dims[l], 1)

		# print(parameters['W1'].shape)
		# print(parameters['W2'].shape)
		return parameters






		




	def linear_forward(self, A_prev, W, b):
		# A_prev = A_prev
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

		# print(W.shape)
		# print(A_prev.shape)

		assert A.shape == (W.shape[0], A_prev.shape[1])

		cache = (linear_cache, activation_cache)
		return A, cache


	def L_model_forward(self, X):
		# print('guagua'+str(X.shape))
		# print(X.shape)
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
		# print(X.shape)
		# print('here')
		# print('AL.shape'+str(AL.shape))
		# print('here')
		# print((X.shape[0],1))
		# print('here')
		assert AL.shape == (10,1)
		return AL, caches


	# Compute cross-entropy cost
	def compute_cost(self,AL, y):
		# y = (np.array(convert_digit_to_array(y)))[np.newaxis,:]
		# print('here')
		# print(y)
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

		# elif activation_fn == "relu":
		# 	dZ = relu_gradient(dA, activation_cache)
		# 	dA_prev, dW, db = linear_backward(dZ, linear_cache)

		return dA_prev, dW, db


	def L_model_backward(self,AL, y, caches, hidden_layers_activation_fn="sigmoid"):
		# y = (np.array(convert_digit_to_array(y)))[np.newaxis,:]

		y = y.reshape(AL.shape)
		L = len(caches)
		# print('here')
		# print(L)
		grads = {}
		A = AL
		y = y
		dAL = np.divide(AL - y, np.multiply(AL, 1 - AL))#10*1
		# print('here')
		# print(dAL.shape)


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



	# def __init__(self, layers_dims, learning_rate, num_iterations,
	# 		print_cost, hidden_layers_activation_fn):

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
			# print('AL size'+ str(AL.shape))
			# print('y size'+ str(y.shape))

			grads = self.L_model_backward(AL, y, caches, self.hidden_layers_activation_fn)

			# update parameters
			parameters = self.update_parameters(parameters, grads, self.learning_rate)

			# append each 100th cost to the cost list
			# if (i + 1) % 100 == 0 and print_cost:
			# 	print(f"The cost after {i + 1} iterations is: {cost:.4f}")

			if i % 10000 == 0:
				cost_list.append(cost)

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
		# output_hidden, cache = lin_forward(input_vector, self.weights_in_hidden)
		# target_vector = (np.array(convert_digit_to_array(label)))[np.newaxis,:]#(1,10)
		# output_hidden_act, cache = lin_activation_forward(input_vector, self.weights_in_hidden, 'sigmoid')
		# output_network_act, cache = lin_activation_forward(output_hidden_act, self.weights_hidden_out, 'softmax')
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
	# labels = np.array(labels)[np.newaxis,:]
	# newlabels=[]
	# print(images.shape)
	# print(labels.shape)
	# print(len([784, 5, 10]))
	model1 = NN([784, 5, 10], 0.1, 1, True, "sigmoid")
	# print(images[0])
	# assert False


	c = (np.array(images[0])[np.newaxis,:])/255
	d = convert_digit_to_array(labels[0])
	# # print(a.shape)
	# model1.L_layer_model(a.T, b)


	i = 0 
	for image,label in zip(images,labels):#60000
		a = (np.array(images[i])[np.newaxis,:])/255
		b = convert_digit_to_array(labels[i])
		# print(a.shape)
		model1.L_layer_model(a.T, b)
		i += 1


	# print(model1.accuracy(c.T,d))
	i = 0
	for image,label in zip(images,labels):
		c = (np.array(images[i])[np.newaxis,:])/255
		d = convert_digit_to_array(labels[i])
		model1.accuracy(c.T,d)
		# print(a.shape)
		# print(model1.accuracy(c.T,d))
		i += 1

	# print(images.shape)
	# print(labels.shape)
	# assert  images.shape == (1,784, 60000) and labels.shape == (10,1, 60000)
	# images = np.array(images)[np.newaxis,:]
	# labels = convert_digit_to_array(labels)[np.newaxis,:]
	# print(labels.shape)
	# print(images.shape)
	# print(labels.shape)


	# print(images[0].shape)

	# model1 = L_layer_model(images, labels, [784, 5, 10], learning_rate=0.01, num_iterations=3000,
	# 		print_cost=True, hidden_layers_activation_fn="sigmoid")
	# print(model1.accuracy(images,model1.parameters,labels,"sigmoid"))


















	# plt.figure(figsize=(28, 28))
	# plt.imshow(images[0])
	# plt.axis("off");
	# test_network = NeuralNetwork(no_of_in_nodes=784, no_of_out_nodes=10, no_of_hidden_nodes=5, learning_rate=0.1)

	# print('before train:'  )
	# print( test_network.run(images[0]))
	# print(labels[0])
	# print(test_network.run(images[1]))
	# print(labels[1])
	# # print(NeuralNetwork.weights_hidden_out)
	# # i=100
	# print(test_network.weights_hidden_out)
	# # for (image,label) in zip(images,labels):
	# # 	test_network.train(image,label)

	# for i in range (0,1):
	# np.random.shuffle((images,labels))
	# for (image,label) in zip(images,labels):
	# 	test_network.train(image,label)


	# i= i-1
	# if (i<=0) :
	# 	break
		
	# print(test_network.run(images[0]))
	# print(labels[0])
	# for (image,label) in zip(images,labels):
	# 	print(locate_max(test_network.run(image)) == label)
	# 	print (locate_max(test_network.run(image)))
	# 	print(label)


	# for (image,label) in zip(images,labels):
	# 	print(test_network.run(image))
	# 	print(label)
	# # 	# print(image)
	# print( images[0] == images[1])
	# print( test_network.run(images[0]) == test_network.run(images[1]))
	# print('after train:' )
	# print(  test_network.run(images[0]))
	# print(labels[0])

	# print(test_network.run(images[1]))
	# print(labels[1])
	# print(test_network.weights_in_hidden)
	# print(test_network.weights_hidden_out)
	# print(judge([0,0.1,0,0,0,0,0,0,0,0.5,]))
	# print(test_network.weights_hidden_out)
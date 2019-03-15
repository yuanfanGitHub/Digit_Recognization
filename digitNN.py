from mnist.loader import MNIST
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from scipy.stats import truncnorm
def activation_function(x):
	return softmax(x)

def sigmoid(x):
	return 1 / (1 + np.e ** -x)
def softmax(x):
	e = np.exp(x - np.max(x))  # prevent overflow
	if e.ndim == 1:
	    return e / np.sum(e, axis=0)
	else:  
		return e / np.array([np.sum(e, axis=1)]).T # ndim = 2
def ReLU(x):
	return x * (x > 0)
def convert_digit_to_array(x):
	a = np.array([0]*10)
	a[x] = 1
	a = np.array(a, ndmin = 1)
	return a
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



mndata = MNIST('./Digitsets')
mndata.gz = True
(images, labels) = mndata.load_training()
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






class NeuralNetwork:
	def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes, learning_rate):
		self.no_of_in_nodes = no_of_in_nodes
		self.no_of_out_nodes = no_of_out_nodes
		self.no_of_hidden_nodes = no_of_hidden_nodes
		self.learning_rate = learning_rate
		self.creat_weights_matrix()

	def creat_weights_matrix(self):
		# rad = 1 / np.sqrt(self.no_of_in_nodes)
		# X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
		# self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, 
		#                                self.no_of_in_nodes))
		# rad = 1 / np.sqrt(self.no_of_hidden_nodes)
		# X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
		# self.weights_hidden_out = X.rvs((self.no_of_out_nodes, 
		#                                 self.no_of_hidden_nodes))
		self.weights_in_hidden = np.random.normal(0,1,size=(self.no_of_in_nodes, self.no_of_hidden_nodes))
		self.weights_hidden_out = np.random.normal(0,1,size=(self.no_of_hidden_nodes, self.no_of_out_nodes))

	def run(self, input_vector):
		input_vector1 = (np.array(input_vector))[np.newaxis,:] /256
		# print(input_vector1.shape)  #(1,784)
		output_vector2 = np.dot(input_vector1, self.weights_in_hidden)
		# print(output_vector2.shape)  #(1,5)
		output_vector3 = activation_function(output_vector2)
		output_vector4 = np.dot(output_vector3, self.weights_hidden_out)
		# print(output_vector4.shape) #(1,10)
		output_vector5 = activation_function(output_vector4)
		# print(judge(output_vector4))
		return output_vector5

	def train(self, input_vector, label):
		input_vector1 = (np.array(input_vector)[np.newaxis,:])/256#(1,784)/256
		# target_vector = np.array(target_vector, ndmin = 1)
		# convert_digit_to_array
		# print(label)
		target_vector1 = (np.array(convert_digit_to_array(label)))[np.newaxis,:]#(1,10)
		# target_vector = activation_function(target_vector)

		output_vector1 = np.dot(input_vector1, self.weights_in_hidden)
		output_hidden = activation_function(output_vector1)
		output_vector2 = np.dot(output_hidden, self.weights_hidden_out)
		output_vector3 = output_vector2
		# print(output_vector3)
		output_network = activation_function(output_vector3)
		# print(output_network)
		# output_network = output_vector2
		# print(output_network)
		# output_network_normalized = exp_normalize(output_network)
		# output_network = output_network_normalized
		# output_errors =  target_vector - output_network
		output_errors =   -output_network + target_vector1
		# errors_out = (target_vector - output_network) * (target_vector - output_network) * 0.5
		#squared error
		#update weights 
		# tmp = np.multiply(output_errors , np.multiply(output_network , (1.0 - output_network)))
		tmp = output_errors * output_network * (1.0 - output_network)
		# tmp = tmp[np.newaxis,:]

		# output_hidden = output_hidden[:,np.newaxis]
		# np.resize(output_hidden,(5,1))
		# print(output_hidden.shape)
		# np.resize(tmp,(1,10))
		# tmp1 = self.learning_rate  * np.dot(output_hidden[:,np.newaxis],tmp)
		tmp1 = self.learning_rate  * np.dot(output_hidden.T, tmp)
		# print(np.size(tmp1))
		self.weights_hidden_out -= tmp1
		# print(self.weights_hidden_out)


		hidden_errors = np.dot(output_errors, self.weights_hidden_out.T)
		# print(np.size(hidden_errors))
		# tmp2 = np.multiply(hidden_errors, np.multiply(output_hidden , (1.0 - output_hidden)))
		tmp2 = hidden_errors * output_hidden * (1.0 - output_hidden)
		# print(np.size(1.0 - output_hidden))
		# print(np.size(tmp2)) #100
		# self.weights_in_hidden -= self.learning_rate * np.dot((input_vector[np.newaxis,:]).T, tmp2[np.newaxis,:])
		self.weights_in_hidden -= self.learning_rate * np.dot((input_vector1).T, tmp2)
		# print(np.size(np.multiply(input_vector.T,tmp2)))



		




test_network = NeuralNetwork(no_of_in_nodes=784, no_of_out_nodes=10, no_of_hidden_nodes=5, learning_rate=0.9)
print('before train:'  )
print( test_network.run(images[0]))
print(labels[0])
print(test_network.run(images[1]))
print(labels[1])
# print(NeuralNetwork.weights_hidden_out)
# i=100
print(test_network.weights_hidden_out)
# for (image,label) in zip(images,labels):
# 	test_network.train(image,label)

for i in range (0,1):
	# np.random.shuffle((images,labels))
	for (image,label) in zip(images,labels):
		test_network.train(image,label)


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
print('after train:' )
print(  test_network.run(images[0]))
print(labels[0])

print(test_network.run(images[1]))
print(labels[1])
# print(test_network.weights_in_hidden)
# print(test_network.weights_hidden_out)
# print(judge([0,0.1,0,0,0,0,0,0,0,0.5,]))
# print(test_network.weights_hidden_out)
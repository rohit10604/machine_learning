from array import array
import csv
import math
import numpy as np
train_data = open('train_data.csv','r')
reader = csv.reader(train_data)
num_neuron_input = 16
num_neuron_output = 26
num_neuron_hidden = 20
W_hidden = []
W_output = []

def sigmoid(x):
  return 1.0 / (1 + math.exp(-x))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

b1= math.sqrt(6.0/(num_neuron_input+num_neuron_hidden))
b2 = math.sqrt(6.0/(num_neuron_output+num_neuron_hidden))
for _ in range(num_neuron_hidden):
	s = np.random.uniform(-b1,b1,num_neuron_input)
	#s.tolist()
	W_hidden.append(s)
#weight matrix for hidden layer generated with xavier's initialisation
for _ in range(num_neuron_hidden):
	s = np.random.uniform(-b2,b2,num_neuron_hidden)
	#s.tolist()
	W_output.append(s)
#weight matrix for output layer generated with xavier's initialisation
#print W_hidden
#print W_output
for row in reader:
	#hidden_layer_values =np.dot(row)
	h1=[]
	o1=[]
	y = np.asarray(row)
	z = y.astype(np.float)
	#hidden_layer_values =np.dot(z,W_hidden)
	for eacharray in W_hidden:
		h1.append(sigmoid(np.dot(z,eacharray)))
	for everyarray in W_output:
		o1.append((np.dot(h1,everyarray)))
	o2 = softmax(o1)
	print "h1"
	print h1
	print "o1"
	print o1 
	print "o2"
	print o2
	break



















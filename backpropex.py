"""
Purpose: This is a backpropagation neural network that is used to classify breast cancer tumors as malignant or benign.
The data set is from the UCI Machine Learning Repository and can be found at https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29.
"""

# -*- coding: utf-8 -*-
from ftplib import all_errors
from random import random
import csv
import math
from math import exp
import networkx as nx
import matplotlib.pyplot as plt

# Initialize a neural network
def initialize_neuralNetwork(inputNeurons, hiddenNuerons, outputNeurons):
	neuralNetwork = list() 
	#neuralNetwork list of layers
	hidden_layer = [{'weights':[random() for i in range(inputNeurons + 1)]} for i in range(hiddenNuerons)]
	#random weights for all the inputs and a weight of 1 on every layer
	neuralNetwork.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(hiddenNuerons + 1)]} for i in range(outputNeurons)]
	#random weights for all the hidden perceptrons and a weight of 1 on every layer
	neuralNetwork.append(output_layer)
	return neuralNetwork 

# Calculate neuron activation for an input
def activate(weights, inputs): 
	#weights is a list of weights for each input
	activation = weights[-1] 
	#this gets the bias
	for i in range(len(weights)-1): 
		#this gets the weights
		activation += weights[i] * inputs[i] 
		#	this is the summation of the weights and inputs
	return activation 
	#this is the activation of the neuron
 
# Transfer neuron activation (change this to change the activation function)
def transfer(activation): 
	#sigmoid function
	return 1.0 / (1.0 + exp(-activation)) 

# def transfer(activation): 
#   	#sigmoid function
# 	return math.tanh(activation)
 
# Forward propagate input to a neuralNetwork output
def forward_propagation(neuralNetwork, row): 
	#row is a list of inputs
	inputs = row 
	for layer in neuralNetwork: 
		prev_outputs = [] 
		for neuron in layer: 
			activation = activate(neuron['weights'], inputs) 
			neuron['output'] = transfer(activation) 
			#output is the sigmoid of the activation
			prev_outputs.append(neuron['output']) 
		inputs = prev_outputs 
		#these are the outputs of the previous layer
	return inputs 
 
# Backpropagate error and store in neurons
def back_propagation(neuralNetwork, expected): 
	#expected is a list of the expected outputs
	for i in reversed(range(len(neuralNetwork))): 
		#this is the output layer
		layer = neuralNetwork[i] 
		errors = list() 
		if i != len(neuralNetwork)-1: 
			#if it is not the output layer
			for j in range(len(layer)): 
				#for each neuron in the layer
				error = 0.0
				for neuron in neuralNetwork[i + 1]: 
					#for each neuron in the next layer
					error += (neuron['weights'][j] * neuron['delta']) 
					#this is the error of the neuron
				errors.append(error)
		else: 
			#if it is the output layer
			for j in range(len(layer)): 
				#for each neuron in the layer
				neuron = layer[j]
				errors.append(neuron['output'] - expected[j]) 
				#this is the error of the neuron
		for j in range(len(layer)): 
			#for each neuron in the layer
			neuron = layer[j]
			neuron['delta'] = errors[j] * (neuron['output'] * (1.0 - neuron['output'])) #this is the delta of the neuron
 
# Update neuralNetwork weights with error
def weight_calc(neuralNetwork, row, l_rate): #row is a list of inputs
	for i in range(len(neuralNetwork)): 
		inputs = row[:-1] 
		if i != 0: #if it is not the input layer
			inputs = [neuron['output'] for neuron in neuralNetwork[i - 1]] 
		for neuron in neuralNetwork[i]: 
			#for each neuron in the layer
			for j in range(len(inputs)): 
				#for each input
				neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j] 
				#this is the weight update
			neuron['weights'][-1] -= l_rate * neuron['delta'] 
			#this is the bias update

# Train a neuralNetwork for a fixed number of epochs
def train_neuralNetwork(neuralNetwork, train, l_rate, n_epoch, neuron_outputs): #
	for epoch in range(n_epoch): 
		#for each epoch
		total_error = 0 
		#this is the sum of the errors
		for row in train: 
			#for each row in the training set
			outputs = forward_propagation(neuralNetwork, row) 
			#this is the output of the neuralNetwork
			expected = [0 for i in range(neuron_outputs)] 
			#instantiating the expected output
			expected[int(row[-1])] = 1
			total_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))]) 
			#this is the error of the neuralNetwork
			back_propagation(neuralNetwork, expected) 
			#this is the backpropagation
			weight_calc(neuralNetwork, row, l_rate) 
			#this is the weight update
		print('Epoch cycle = %d, learning-rate= %.2f, error = %.4f' % (epoch, l_rate, total_error)) #



data = open("wdbc.data", "r")
reader = csv.reader(data, delimiter=',')
dataset = list(reader)


#every 32 rows is a new patient

for i in range(len(dataset)):
	if dataset[i][1] == 'M':
		# M = 1
		dataset[i][1] = 1
	else:
		dataset[i][1] = 0
		# B = 0
for row in dataset:
	del row[0]



for i in range(len(dataset)):
	#for each row
	for j in range(0, len(dataset[i])):
		#for each column
		if j != 0 :
			dataset[i][j] = float(dataset[i][j])
			#convert to float

# print(dataset)

n_inputs = len(dataset[0]) - 1    
#number of inputs 32
neuron_outputs = len(set([row[0] for row in dataset])) 
neuralNetwork = initialize_neuralNetwork(n_inputs, 4, neuron_outputs) 
#hidden layer has 4 neurons
train_neuralNetwork(neuralNetwork, dataset, 0.5, 20, neuron_outputs) 
#learning rate = 0.5, 20 epochs
for layer in neuralNetwork:
	print("\n")
	print(layer)
	print("\n")






def plot_neural_network(neural_network):
    # Create a graph object
    graph = nx.DiGraph()

    # Add nodes and edges for each layer in the neural network
    layer_positions = {}
    max_neurons_in_a_layer = 0

    for layer_idx, layer in enumerate(neural_network):
        max_neurons_in_a_layer = max(max_neurons_in_a_layer, len(layer))
        for neuron_idx, neuron in enumerate(layer):
            neuron_id = f"Layer {layer_idx + 1} - Neuron {neuron_idx + 1}"
            graph.add_node(neuron_id)
            layer_positions[neuron_id] = (layer_idx, -neuron_idx)
            if layer_idx > 0:
                # Connect to previous layer
                for prev_neuron_idx, _ in enumerate(neural_network[layer_idx - 1]):
                    prev_neuron_id = f"Layer {layer_idx} - Neuron {prev_neuron_idx + 1}"
                    graph.add_edge(prev_neuron_id, neuron_id)

    # Generate plot positions
    pos = {}
    layer_gap = 1.0
    neuron_gap = 2.0 / max_neurons_in_a_layer

    for neuron_id, (layer_idx, neuron_idx) in layer_positions.items():
        pos[neuron_id] = (layer_idx * layer_gap, neuron_idx * neuron_gap)

    # Draw the network
    nx.draw(graph, pos, node_color="skyblue", with_labels=True, node_size=3000, font_size=8, font_color="black", font_weight="bold")
    plt.title("Neural Network Architecture")
    plt.show()

# The rest of your code stays the same

# Initialize and train your neural network as you did before

# Add this line after you have trained the neural network:
plot_neural_network(neuralNetwork)
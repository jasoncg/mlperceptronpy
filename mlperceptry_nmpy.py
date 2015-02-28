# jasoncg
#
# mlperceptron_nmpy.py - A multilayer perceptron implementation in Python
# Unlike mlperceptron.py, this implementation uses matrices to optimize computation
#
import timer
import random
import math
import numpy as np

def ndprint(a, format_string ='{0:.2f}'):
	for i in range(0, len(a)):
		print("%s\n%s" %(i, a[i]))
	#for i,v in enumerate(a):
	#	print([format_string.format(v,i)])
	#print([format_string.format(v,i) for i,v in enumerate(a)])


class Neuron:
	@staticmethod
	def tanh(input):
		return np.tanh(input)
	@staticmethod
	def dxtanh(input):
		return 1.0-np.power(Neuron.tanh(input), 2.0)
	'''
	@staticmethod
	def sigmoid(input):
		try:
			ex=math.exp(-input)
			return 1.0/(1.0+ex)
		except:
			# If overflow, snap to either 1 or 0
			if -input>0:
				# lim(sigmoid(x), -inf)=0
				return 1
			else:
				# lim(sigmoid(x), inf)=1
				return 0
			#print("Fatal Error %s", -input)

	@staticmethod
	def dxsigmoid(input):
		#ex = math.exp(input)
		#return (ex/math.pow(1+ex, 2.0))
		s = Neuron.sigmoid(input)
		return s*(1-s)
	@staticmethod
	def activate(input):
		return Neuron.sigmoid(input)
	@staticmethod
	def dxactivate(input):
		return Neuron.dxsigmoid(input)

	@staticmethod
	def to_output(output):
		if output<0.5:
			return 0
		return 1
	'''
	@staticmethod
	def activate(input):
		return Neuron.tanh(input)
	@staticmethod
	def dxactivate(input):
		return Neuron.dxtanh(input)

	@staticmethod
	def to_output(output, min=0.0, max=1.0, round=True):
		#range = max-min
		#r2 = range/2.0
		result = output*((max-min)/2.0)+(max+min)/2.0

		#print("Output[%s]\nResult[%s]"%((output), (result)))
		#print("to_output %s %s :: %s :: result %s"% (min, max, output, result))
		if round:
			# If around 0.5, force round up
			return np.round(result+0.0000000001)
		return result
		'''
		if output<=0.0:
			return 0
		return 1
		'''
	'''
	@staticmethod
	def evaluate(weights, inputs):
		# add for bias
		inputs=np.insert(inputs, 0, 1.0)
		#print("Evaluate %s %s\n\n" % (len(weights), len(inputs)))
		result = 0
		#for weight_index in range(0, len(weights)):
		#	result+=inputs[neuron_index]*weights[weight_index]

		result = np.dot(weights,inputs)

		
		return Neuron.activate(result)
	'''

	@staticmethod
	def evaluaten(neurons, inputs):
		# add for bias
		inputs=np.insert(inputs, 0, 1.0)
		#print("Evaluate %s %s\n\n" % (len(weights), len(inputs)))
		#for weight_index in range(0, len(weights)):
		#	result+=inputs[neuron_index]*weights[weight_index]

		result = np.dot(neurons, inputs)

		return Neuron.activate(result)

	@staticmethod
	def backpropagate(last_guessed, err):
		di = Neuron.dxactivate(last_guessed)
		#print('backprop\n%s\n%s\n%s'%(last_guessed, err, di))
		return np.multiply(err, di)

	@staticmethod
	def calculate_error(last_guessed, expected, min=0.0, max=1.0, round=True):
		#Scale expected values for proper range
		
		#result = output*((max-min)/2.0)+(max+min)/2.0
		#return expected - last_guessed

		expected_scaled = (expected - (max+min)/2.0)/((max-min)/2.0)


		#print("calcerror %s %s :: %s"% (expected, expected_scaled, last_guessed))
		#return expected_scaled - last_guessed
		diff = expected_scaled - last_guessed
		return diff
		'''
		c = np.divide(np.power(last_guessed - expected_scaled, 2.0), 2.0) #-2.0
		#c = np.power(last_guessed - expected_scaled, 2.0)

		print("Computed[%s] Target[%s]\n%s\n%s\n\n"% (
			last_guessed, expected_scaled,
			diff, 
			c))
		return c
		'''
	#def backpropagate(self, err):
	#	di = Neuron.dxactivate(self.last_guessed)
	#	self.error = err * di
	#	return self.error

	# Updates weights given the old weights, error, and inputs
	# Returns new weights
	@staticmethod
	def backpropagate_update_weights(weights, error, inputs, learning_rate):
		
		#inputs=np.insert(inputs, 0, 1.0)
		#print("Weights\n%s\n+\nError\n%s\n*\nInputs\n%s"%(weights,error, inputs))
		change = np.multiply(np.rot90([learning_rate*error], -1), (inputs))
		'''
		if len(error)>1:
			print("backpropagate_update_weights")
			print([learning_rate*error])
			print("ROT=>")
			print(np.rot90([learning_rate*error], -1))
			print(" *Inputs ")
			print(inputs)
		if len(error)>1:
			print(" =Change ")
			print(change)
			print(" +Weights ")
			print(weights)
			print(" =NewWeights ")
			print(np.add(weights, change))
		'''

		#print("Weights\n%s\n+\Change\n%s"%(weights,change))
		return np.add(weights, change)
	'''
	def backpropagate_update_weights(self, inputs, learning_rate):
		
		inputs2=np.insert(inputs, 0, 1.0)

		change = np.multiply(inputs2, learning_rate*self.error)

		self.weights=np.add(self.weights, change)
		return self.evaluate(inputs)
	'''
	'''
	# First, update the bias
	self.weights[0] = self.weights[0] + learning_rate*self.error
	for i in range(1, len(self.weights)):
		change = learning_rate*self.error*inputs[i-1]

		self.weights[i] = self.weights[i] + change
	return self.evaluate(inputs)
	'''
'''
A Multilayer Perceptron implementation
Each layer is stored as a matrix (2d array) in self.layers.  The first row is the input 
layer, the last is the output layer, and any other layers are hidden layers.

Each row in the matrix represents a neuron.  Each neuron within a layer has the same 
number of weights, which coresponds to the number of neurons in the previous layers
(or the number of inputs in the input layer).

The actual values indicate the weights making up that layer.  The number of
neurons in a layer equals the length of the layer / number of neurons in 
the previous layer.
'''
class Perceptron:
	@staticmethod
	def new_weight(output_range=[0.0, 1.0]):
		# Generate a random value in the output range
		#r= random.uniform(output_range[0], output_range[1])#-(output_range[1]-output_range[0])/2.0
		#return r
		return random.random()*2.0-1.0
	@staticmethod
	def generate_weights(input_count, neuron_count, output_range=[0.0, 1.0]):
		#length = neuron_count*(input_count+1)
		# Add one for bias
		input_count+=1
		results = None
		for n in range(0, neuron_count):
			neuron = np.zeros(input_count, np.float32)
			for i in range(0, input_count):
				neuron[i] = Perceptron.new_weight(output_range)
			if results is None:
				results = np.array([neuron])
			else:
				results = np.vstack((results, neuron))
		return results

	@staticmethod
	def new_perceptron_random(input_count, neuron_count, output_range=[0.0, 1.0], output_integer=True):
		neurons = Perceptron.generate_weights(input_count, neuron_count, output_range)
		return Perceptron(input_count, neurons, output_range, output_integer)
	'''
	def get_weight_count(self):
		c=0
		for n in range(0, len(self.neurons)):
			c+=len(self.neurons[n].weights)
		return c
	'''

	def __init__(self, input_count, neurons, output_range=[0, 1], output_integer=True):
		self.input_count = input_count
		self.layers=[]
		#self.layers_weightsperneuron=[]
		#self.layers_neuroncount=[]
		self.output_range = output_range
		self.output_integer = output_integer
		if not isinstance(neurons, np.ndarray):
			neurons = np.array(neurons)

		self.layers.append(neurons)
		#self.layers_weightsperneuron.append(input_count)
		# For input layer, input_count == neuron_count
		#self.layers_neuroncount.append(len(neurons))

	def add_next_layer(self, neuron_count, weights=None):
		new_layer_index = len(self.layers)
		weights_per_neuron = len(self.layers[new_layer_index-1])
		# Get neuron count for layer
		w = weights
		if w==None:
			# input_count[l] = neuron_count[l-1]
			w=Perceptron.generate_weights(weights_per_neuron, neuron_count)
		
		if not isinstance(weights, np.ndarray):
			weights = np.array(weights)

		self.layers.append(w)
		#self.layers_weightsperneuron.append(weights_per_neuron)
		#self.layers_neuroncount.append(neuron_count)

		return self
	'''
	def renormalize(self):
		# Resets all weights to be within output_range
		#scale = (self.output_range[1]-self.output_range[0])*2.0
		#scale_half= scale/2.0
		scale = 2.0
		scale_half = 1.0

		min_value = None
		max_value = None

		# Find min and max
		for l in range(0, len(self.layers)):
			mn = np.min(self.layers[l])
			mx = np.max(self.layers[l])
			if min_value is None or mn<min_value:
				min_value = mn
			if max_value is None or mx>max_value:
				max_value = mx

		for l in range(0, len(self.layers)):
			self.layers[l]=np.divide(
							np.subtract(self.layers[l], min_value),
							max_value-min_value)*scale - scale_half
	'''


	def grow(self, layer_index, neuron_count=1):
		# Add the specified number of neurons to the specified layer
		# (note: also adds corresponding weights to the neurons in the next layer)
		if layer_index==0:
			input_count = self.input_count
		else:
			input_count = len(self.layers[layer_index-1])
		old_layer_size = len(self.layers[layer_index])
		# Add them to the layer
		#print("OLD")
		#ndprint(self.layers)
		#self.renormalize()

		#self.layers[layer_index] = np.vstack((self.layers[layer_index], new_neurons))
		# Step through each neuron and add it to the layer
		for n in range(0, neuron_count):
			# Generate the new neurons
			new_neurons = Perceptron.generate_weights(input_count, 1, self.output_range)
			#print("New Neurons %s" % new_neurons)
			self.layers[layer_index] = np.insert(self.layers[layer_index], 
													len(self.layers[layer_index]),
													0, #new_neurons[0],
													axis=0)
		# Add corresponding weights to the neurons in the following layer, if applicable (not output layer)
		if layer_index<=len(self.layers)-1:
			new_number_weights = len(self.layers[layer_index])
			#print("%sx%s %s" % (new_number_weights, len(self.layers[layer_index+1]), self.layers[layer_index+1]))
			#self.layers[layer_index+1] = np.resize(self.layers[layer_index+1], (
			#	len(self.layers[layer_index+1]),	# number of neurons in next layer no change
			#	new_number_weights+1				# number of weights in next layer = number of neurons in this layer + 1 bias
			#	))
			# add new weights to the next layer
			#print("OLD %s" % self.layers[layer_index+1])
			for n in range(0, neuron_count):
				#new_weights = Perceptron.generate_weights(old_layer_size+1+n, 1)
				self.layers[layer_index+1] = np.insert(self.layers[layer_index+1], 
														old_layer_size+1+n, 
														1, 
														axis=1)
			#print("NEW %s" % self.layers[layer_index+1])
			#print(self.layers[layer_index+1])
			# step through each neuron
			#for neuron_index in range(0, len(self.layers[layer_index+1])):
			#	for w in range(len(self.layers[layer_index-1])-neuron_count+1, len(self.layers[layer_index-1]+1)):
			#		#weight = Perceptron.new_weight()
			#		self.layers[layer_index+1][neuron_index][w]=0
		#print("NEW")
		#ndprint(self.layers)

	def evaluate_error(self, layer_index, inputs, expected):
		results = self.evaluate(layer_index, inputs)
		error = 0
		for i in range(0, len(expected)):
			if results[i]!=expected[i]:
				error+=1
		error=error/len(expected)

		return error
	# Evaluate the given inputs
	# If dropout_rate>0 then a dropout mask is applied to the weights in each
	# hidden layer with a percentage of each layer's weights marked out so as
	# not to be used in the evaluation.  This mask is saved for backpropagation
	# so those masked weights are not updated.
	def evaluate(self, inputs, layer_index=0, cache_results=False, dropout_rate=0.0, normalized=False):
		if not normalized and layer_index==0:
			# Only normalize the input layer
			inputs = Perceptron.normalize_data(inputs, self.output_range[0],self.output_range[1])

		# Reset cache_results if applicable
		if layer_index==0:
			if cache_results:
				self.last_results=[]
			else:
				self.last_results = None
			if dropout_rate>0.0:
				self.evaluate_dropout=len(self.layers)*[[]]
			else:
				self.evaluate_dropout=None
			#print("evaluate_dropout %s %s" % (dropout_rate, self.evaluate_dropout))

		#results = np.zeros(len(self.neurons), np.float32)#zeros_like(len(self.neurons)*[0.0]) #len(self.neurons)*[0]
		results = np.zeros(len(self.layers[layer_index]), np.float32)
		layer_dropout_rate=0.0
		neurons_count = len(self.layers[layer_index])

		# If this is a hidden layer, apply dropout (if set)
		if layer_index>0 and layer_index<len(self.layers)-1:
			#if neurons_count>2:
			layer_dropout_rate = np.floor(neurons_count*dropout_rate)

		
		if self.evaluate_dropout is not None:
			#if layer_dropout_rate>0.0:
			# Initially, use all neurons in this layer
			self.evaluate_dropout[layer_index]=(neurons_count*[1])
			#else:
			#	self.evaluate_dropout[layer_index]=None

		#print('Evaluate Layer %s' % layer_index)
		
		if layer_dropout_rate>0:
			# Pick the neurons to be dropped out
			#print("DroputRate %s*%s %s\n" %(neurons_count, dropout_rate, drop_neurons))
			dropout_mask_indices = random.sample(range(neurons_count), np.int(layer_dropout_rate))
			for i in dropout_mask_indices:
				self.evaluate_dropout[layer_index][i]=0.0
			#print("DroputMask%s\n%s\n"%(layer_index, self.evaluate_dropout[layer_index]))
		#print(self.layers)
		neurons = self.layers[layer_index]
		if layer_dropout_rate>0:
			#print("EVALUATE DROPOUT====================")
			#print(neurons)
			#print("*")
			#print(self.evaluate_dropout[layer_index])
			neurons = np.multiply(neurons, np.rot90([self.evaluate_dropout[layer_index]], -1))
			#print("OUTPUT DROPOUT====================")
			#print(neurons)
		resultsn=Neuron.evaluaten(neurons, inputs)
		for i in range(0, len(resultsn)):
			results[i]=resultsn[i]

		#neuron_weights_count = self.layers_weightsperneuron[layer_index]

		#for neuron_index in range(0, neurons_count):
		#	results[neuron_index]=Neuron.evaluate(self.layers[layer_index][neuron_index], inputs)

		if cache_results:
			self.last_results.append(results)

		if layer_index<len(self.layers)-1:
			# If not output layer, continue
			return self.evaluate(results, layer_index+1, cache_results, dropout_rate)
		else:
			# If output layer, convert to boolean outputs
			#for i in range(0, len(results)):
			#	results[i]=Neuron.to_output(results[i], self.output_range[0], self.output_range[1], self.output_integer)
			resultsn=Neuron.to_output(results, self.output_range[0], self.output_range[1], self.output_integer)
			#for i in range(0, len(resultsn)):
			#	results[i]=resultsn[i]
			results = resultsn
			#if results!=resultsn:
			#	print("ERR: %s %s" %(results, resultsn))
			return results

	# Calculate the error for this layer applicable to the 
	# specified neuron on the previous layer
	# En=Sum(Win*Ei)
	def get_error_for(self, layer_index, previous_layer_neuron_index):
		output = 0
		#if previous_layer_neuron_index is None:
		#	r=np.dot(np.rot90(self.layers[layer_index]),(self.backpropagate_errors[layer_index]))
		#	print("%s\n*\n%s\n=\n%s"%(self.layers[layer_index],self.backpropagate_errors[layer_index],r))
		#	return r

		# Step through each neuron on this layer
		for neuron_index in range(0, len(self.layers[layer_index])):
			#index+1 to account for bias (weight[0] is a bias, not fed by a neuron)
			#if self.neurons[i].dropout!=True:
			#print("%s\n*\n%s\n\n"%(self.layers[layer_index][neuron_index][previous_layer_neuron_index+1],self.backpropagate_errors[layer_index][neuron_index]))
			if self.evaluate_dropout==None or self.evaluate_dropout[layer_index][neuron_index]!=0:
				output+=(self.layers[layer_index][neuron_index][previous_layer_neuron_index+1]
							*self.backpropagate_errors[layer_index][neuron_index])
		return output

	@staticmethod
	def normalize_data(values, min, max):
		# Normalize the dataset to [-1, 1]
		#minp = -1
		#maxp = 1
		#print("normalize_data %s %s\n%s\n" %(min, max, values))
		n = np.divide(
					np.subtract(values, np.float32(min)),
					np.float32(max-min))
		r = np.subtract(np.multiply(n, 2.0), 1.0)
		#print("Normalize\n%s\n =>\n%s\n%s\n" %(values, n, r))
		return r

	# Update the weights using backpropagation.  Returns the claculated value(s)
	# after applying the algorithm with updated weights
	def backpropagate(self, inputs, expected, learning_rate, layer_index=0, normalized=False):
		is_input_layer = False
		is_output_layer = False
		if not normalized and layer_index==0:
			inputs = Perceptron.normalize_data(inputs, self.output_range[0],self.output_range[1])

		if layer_index==0:
			is_input_layer = True
			self.backpropagate_errors=len(self.layers)*[[]]
		if layer_index==len(self.layers)-1:
			is_output_layer = True

		neuron_count = len(self.layers[layer_index])

		if is_input_layer:
			# Input layer
			self.evaluate(inputs, layer_index, True, normalized=True, dropout_rate=0.0)#, 0.25)

		self.backpropagate_errors[layer_index]=np.zeros(neuron_count, np.float32)
		if not is_output_layer:
			# Not output layer
			errs = self.backpropagate(None, expected, learning_rate, layer_index+1)

			for neuron_index in range(0, neuron_count):
				#if self.neurons[neuron_index].dropout==True:
				#	continue
				if self.evaluate_dropout!=None and self.evaluate_dropout[layer_index][neuron_index]==0:
					continue

				err = self.get_error_for(layer_index+1, neuron_index)
				# Pass the last calculated value for this neuron, and the error from the next layer
				self.backpropagate_errors[layer_index][neuron_index] = Neuron.backpropagate(self.last_results[layer_index][neuron_index], err)
				#e = self.neurons[neuron_index].backpropagate(err)
			'''
			errs = self.get_error_for(layer_index+1)
			# Pass the last calculated value for this neuron, and the error from the next layer
			self.backpropagate_errors[layer_index] = Neuron.backpropagate(self.last_results[layer_index], errs)
			#for i in range(0, len(resultsn)):
			#	results[i]=resultsn[i]
			'''
			
		else:
			# Output layer
			total_errors = 0
			errs = Neuron.calculate_error(self.last_results[layer_index], expected, self.output_range[0], self.output_range[1], self.output_integer)

			resultsn = Neuron.backpropagate(self.last_results[layer_index], errs)
			for neuron_index in range(0, neuron_count):
				#print("NI %s"%neuron_index)
				#print("%s %s" % (resultsn, (self.backpropagate_errors[layer_index])))
				self.backpropagate_errors[layer_index][neuron_index]=resultsn[neuron_index]
			return errs
			'''
			self.backpropagate_errors[layer_index]=np.zeros(neuron_count, np.float32)
			for neuron_index in range(0, neuron_count):
				err = Neuron.calculate_error(self.last_results[layer_index][neuron_index], expected[neuron_index])

				self.backpropagate_errors[layer_index][neuron_index] = Neuron.backpropagate(self.last_results[layer_index][neuron_index], err)

				lg = Neuron.to_output(self.last_results[layer_index][neuron_index])
				if lg!=expected[neuron_index]:
					total_errors+=1
			'''
			# Adjust learning_rate
			#learning_rate*=(1-total_errors/neuron_count)
			#if learning_rate>1.0:
			#	learning_rate=1.0
			#elif learning_rate<=0.0001:
			#	learning_rate=0.0001
		# Now update the weights

		if is_input_layer:
			# Back to input layer
			return self.backpropagate_update_weights(
				layer_index, inputs, learning_rate
				)
			#self.backpropagate_update_weights(layer_index, inputs, learning_rate)
		#print(np.sum(errs))
		#return errs

	def backpropagate_update_weights(self, layer_index, inputs, learning_rate):
		# Add bias to input
		inputs2=np.insert(inputs, 0, 1.0)

		#results = np.zeros(self.layers_neuroncount[layer_index], np.float32)
		#for neuron_index in range(0, self.layers_neuroncount[layer_index]):
		#	if self.evaluate_dropout!=None and self.evaluate_dropout[layer_index][neuron_index]==True:
		#		continue
		self.layers[layer_index] = Neuron.backpropagate_update_weights(
										self.layers[layer_index], 
										self.backpropagate_errors[layer_index], 
										inputs2, 
										learning_rate)
		#print(self.layers[layer_index])
		#Recalculate this layer to pass onto next layer
		results = Neuron.evaluaten(self.layers[layer_index], inputs)
			#if self.neurons[i].dropout!=True:
			#	results[i]=self.neurons[i].backpropagate_update_weights(inputs, learning_rate)
			#else:
			#	results[i]=0
		if layer_index<len(self.layers)-1:
			return self.backpropagate_update_weights(layer_index+1, results, learning_rate)
		else:
			return results

trainingset=[{'name': 'AND', 
				'inputs': 2, 
				'layers': [1, 1, 1],
				'training': [
					[[0,0], [0]],
					[[0,1], [0]],
					[[1,0], [0]],
					[[1,1], [1]]
				]

			},{'name': 'OR', 
				'inputs': 2, 
				'layers': [1, 1, 1],
				'training': [
					[[0,0], [0]],
					[[0,1], [1]],
					[[1,0], [1]],
					[[1,1], [1]]
				]

			},{'name': 'XOR', 
				'inputs': 2, 
				'layers': [1, 1, 1],
				'training': [
					[[0,0], [0]],
					[[0,1], [1]],
					[[1,0], [1]],
					[[1,1], [0]]
				]

			},{'name': 'NEGATE', 
				'inputs': 1, 
				'output_range': [-2, 2],
				'layers': [1, 1, 1],
				'training': [
					#[[-5],  [ 5]],
					#[[-4],  [ 4]],
					[[-2],  [ 2]],
					[[-1],  [ 1]],
					[[ 0],  [ 0]],
					[[ 1],  [-1]],
					[[ 2],  [-2]],
					#[[ 4],  [-4]],
					#[[ 5],  [-5]]
				],
				#'testing': [
				#	[[-3],  [ 3]],
				#	[[ 3],  [-3]]
				#]

			},{'name': 'ADDER', 
				'inputs': 2, 
				'output_range': [0, 6],
				'output_integer': True,
				'layers': [1, 1, 1],
				'training': [
					[[0,0], [0]],
					[[0,1], [1]],
					[[0,2], [2]],
					[[0,3], [3]],

					[[1,0], [1]],
					[[1,1], [2]],
					[[1,2], [3]],
					[[1,3], [4]],
					
					[[2,0], [2]],
					[[2,1], [3]],
					[[2,2], [4]],
					[[2,3], [5]],
					
					[[3,0], [3]],
					[[3,1], [4]],
					[[3,2], [5]],
					[[3,3], [6]]
				]

			}
]

'''
					[[0, 0], [0, 0, 0]],
					[[0, 1], [0, 0, 1]],
					[[0, 2], [0, 1, 0]],

					[[1, 0], [0, 0, 1]],
					[[1, 1], [0, 1, 0]],
					[[1, 2], [0, 1, 1]],

					[[2, 0], [0, 1, 0]],
					[[2, 1], [0, 1, 1]],
					[[2, 2], [1, 0, 0]],


					[[0,0, 0,0], [0, 0, 0]],
					[[0,0, 0,1], [0, 0, 1]],
					[[0,0, 1,0], [0, 1, 0]],
					[[0,0, 1,1], [0, 1, 1]],

					[[0,1, 0,0], [0, 0, 1]],
					[[0,1, 0,1], [0, 1, 0]],
					[[0,1, 1,0], [0, 1, 1]],
					[[0,1, 1,1], [1, 0, 0]],
					
					[[1,0, 0,0], [0, 1, 0]],
					[[1,0, 0,1], [0, 1, 1]],
					[[1,0, 1,0], [1, 0, 0]],
					[[1,0, 1,1], [1, 0, 1]],
					
					[[1,1, 0,0], [0, 1, 1]],
					[[1,1, 0,1], [1, 0, 0]],
					[[1,1, 1,0], [1, 0, 1]],
					[[1,1, 1,1], [1, 1, 0]]

			[[0,0], [0]],
			[[0,1], [1]],
			[[0,2], [2]],
			[[0,3], [3]],

			[[1,0], [1]],
			[[1,1], [2]],
			[[1,2], [3]],
			[[1,3], [4]],
			
			[[2,0], [2]],
			[[2,1], [3]],
			[[2,2], [4]],
			[[2,3], [5]],
			
			[[3,0], [3]],
			[[3,1], [4]],
			[[3,2], [5]],
			[[3,3], [6]]

'''
iterations_per_epoch= 100
epochs_until_grow	= 2
grow_amount			= 2
def test(sample, learning_rate, silent = False):
	learning_rate = np.float32(learning_rate)
	cp_epochs_until_grow	= epochs_until_grow
	cp_learning_rate		= learning_rate

	with timer.Timer(sample['name']) as t:
		minimum_error=None
		generations_since_minimum_error=0
		grow_count=0
		grow_layer=0

		for i in range(0, len(sample['training'])):
			sample['training'][i][0]=np.array(sample['training'][i][0], np.float32)
			sample['training'][i][1]=np.array(sample['training'][i][1], np.float32)

		if 'testing' in sample:
			for i in range(0, len(sample['testing'])):
				sample['testing'][i][0]=np.array(sample['testing'][i][0], np.float32)
				sample['testing'][i][1]=np.array(sample['testing'][i][1], np.float32)

		if 'output_range' in sample:
			output_range = sample['output_range']
		else:
			output_range = [0, 1]

		if 'output_integer' in sample:
			output_integer = sample['output_integer']
		else:
			output_integer = True

		test_data = list(sample['training'])

		if 'testing' in sample:
			test_data += sample['testing']
		output_size = sample['layers'][-1]
		p=Perceptron.new_perceptron_random(sample['inputs'], sample['layers'][0], output_range, output_integer)
		for i in range(1, len(sample['layers'])):
			p.add_next_layer(sample['layers'][i])
		if not silent:
			t.print_elapsed('INIT')
		
			for tr in sample['training']:
				print("%s=%s ?? Actual = %s"%(tr[0], tr[1], p.evaluate(tr[0])))

			t.print_elapsed("EVAL")

		iterations = 0
		#for i in range(0, 500):
		while True:
			# Shuffle the training set to train in a different order each
			# epoch
			tlist = list(range(0, len(sample['training'])))
			np.random.shuffle(tlist)
			for i in range(0, iterations_per_epoch):
				#print("%s\n%s" %(sample['training'], t))
				for index in tlist:
					p.backpropagate(
						sample['training'][index][0], 
						sample['training'][index][1], 
						np.float32(cp_learning_rate))
			iterations+=iterations_per_epoch

			if not silent:
				t.print_elapsed("Train")
			errors = 0


			for tr in test_data:
				results = p.evaluate(tr[0])
				if not silent:
					print("%s=%s ?? Actual = %s"%(tr[0], tr[1], results))
				for r in range(0, len(results)):
					if tr[1][r]!=results[r]:
						errors+=1

			if not silent:
				print("I = %s\tE = %s (Emin = %s)"% (iterations, errors, minimum_error))

			if errors==0:
				break
			total_output_size = output_size * len(test_data)
			cp_learning_rate=learning_rate*(errors/total_output_size)
			if not silent:
				print("NewLearningRate: %s = (%s*(%s/%s))" %(cp_learning_rate, learning_rate, errors, total_output_size))
			if minimum_error is None or errors<minimum_error:
				minimum_error = errors
				generations_since_minimum_error = 0
			else:
				generations_since_minimum_error+=1
				if generations_since_minimum_error>=cp_epochs_until_grow:
					# Increase epochs until grow by 20%
					cp_epochs_until_grow*=1.5
					#for tr in test_data:
					#	results = p.evaluate(tr[0])
					#	print("PRE %s=%s ?? Actual = %s"%(tr[0], tr[1], results))
					generations_since_minimum_error=0
					minimum_error=None
					p.grow(grow_layer, grow_amount)
					if not silent:
						print("****************\nGrow Layer %s (grow_count==%s)\n****************" % (grow_layer, grow_count+1))
					grow_layer+=1
					if grow_layer==len(p.layers)-1:
						grow_layer = 0
					grow_count+=1
					#for tr in test_data:
					#	results = p.evaluate(tr[0])
					#	print("PGR %s=%s ?? Actual = %s"%(tr[0], tr[1], results))

		if not silent:
			print("Completed within %s iterations\nGrown %s times" % (iterations, grow_count))
			print("******************************************\n******************************************")
			ndprint(p.layers)
			print("******************************************\n******************************************")
	return iterations
#test(trainingset[4], 0.005)

stats = {}

with timer.Timer('Test Suite') as t:
	for count in range(50):
		for i in range(0, len(trainingset)):
			iterations = test(trainingset[i], 0.05, True)
			ellapsed = t.get_elapsed()
			if trainingset[i]['name'] not in stats:
				stats[trainingset[i]['name']] = {
							'min_time': ellapsed, 'max_time': ellapsed, 
							'min_iterations': iterations, 'max_iterations': iterations}
			if iterations<stats[trainingset[i]['name']]['min_iterations']:
				stats[trainingset[i]['name']]['min_iterations'] = iterations
			if iterations>stats[trainingset[i]['name']]['max_iterations']:
				stats[trainingset[i]['name']]['max_iterations'] = iterations

			if ellapsed<stats[trainingset[i]['name']]['min_time']:
				stats[trainingset[i]['name']]['min_time'] = ellapsed
			if ellapsed>stats[trainingset[i]['name']]['max_time']:
				stats[trainingset[i]['name']]['max_time'] = ellapsed

			#print("%s took %s iterations" %(trainingset[i]['name'], iterations))
			print("Statistics for %s"% trainingset[i]['name'])
			for k in stats[trainingset[i]['name']]:
				print("%s\t:\t%s"%(k, stats[trainingset[i]['name']][k]))
			print("***************************************************\n\n")
			#print("%s\n\n" %stats[trainingset[i]['name']])

for s in stats:
	print("Statistics for %s"% s)
	for k in stats[trainingset[i]['name']]:
		print("%s\t:\t%s"%(k, stats[trainingset[i]['name']][k]))
	#print("%s\n\n" %stats[s])

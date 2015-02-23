import random
import math

class Neuron:
	def __init__(self, index, weights, perceptron_layer):
		self.weights = weights
		self.layer = perceptron_layer
		self.last_guessed = None
		self.error = None
		self.dropout = False
	
	
	@staticmethod
	def tanh(input):
		return math.tanh(input)
	@staticmethod
	def dxtanh(input):
		return 1.0-math.pow(Neuron.tanh(input), 2.0)
	@staticmethod
	def sigmoid(input):
		try:
			ex=math.exp(-input)
			return 1.0/(1.0+ex)
		except:
			print("Fatal Error %s", -input)

	@staticmethod
	def dxsigmoid(input):
		ex = math.exp(input)
		return (ex/math.pow(1+ex, 2.0))
	
	@staticmethod
	def activate(input):
		return Neuron.sigmoid(input)
	@staticmethod
	def dxactivate(input):
		return Neuron.dxsigmoid(input)
	def last_guessedb(self):
		if self.last_guessed<0.5:
			return False
		return True
	@staticmethod
	def to_output(output):
		if output<0.5:
			return False
		return True
	'''
	@staticmethod
	def activate(input):
		return Neuron.tanh(input)
	@staticmethod
	def dxactivate(input):
		return Neuron.dxtanh(input)

	def last_guessedb(self):
		if self.last_guessed<=0.0:
			return False
		return True
	@staticmethod
	def to_output(output):
		if output<=0.0:
			return False
		return True
	'''
	def evaluate(self, inputs):
		self.dropout = False
		result = 0

		#Calculate bias
		result+=1*self.weights[0]
		for i in range(1, len(self.weights)):
			result+=inputs[i-1]*self.weights[i]

		self.last_guessed=Neuron.activate(result)

		return self.last_guessed

	def calculate_error(self, expected):
		self.error = expected - self.last_guessed
		return self.error

	def backpropagate(self, err):
		di = Neuron.dxactivate(self.last_guessed)
		self.error = err * di
		return self.error

	def backpropagate_update_weights(self, inputs, learning_rate):
		# First, update the bias
		self.weights[0] = self.weights[0] + learning_rate*self.error
		for i in range(1, len(self.weights)):
			change = learning_rate*self.error*inputs[i-1]

			self.weights[i] = self.weights[i] + change

		return self.evaluate(inputs)

class Perceptron:
	@staticmethod
	def generate_weights(input_count, neuron_count):
		length = neuron_count*(input_count+1)
		results = length*[0] #np.zeros(length)# []
		for i in range(0, length):
			results[i] = random.random()*2.0-1.0
		return results

	@staticmethod
	def new_perceptron_random(input_count, neuron_count, perceptron_layer_prev=None):
		weights = Perceptron.generate_weights(input_count, neuron_count)
		return Perceptron(input_count, neuron_count, weights, perceptron_layer_prev)

	def get_weight_count(self):
		c=0
		for n in range(0, len(self.neurons)):
			c+=len(self.neurons[n].weights)
		return c

	def __init__(self, input_count, neuron_count, weights, perceptron_layer_prev=None):
		self.neurons = neuron_count*[None]
		self.layer_prev = perceptron_layer_prev
		self.layer_next = None

		if self.layer_prev == None:
			self.index = 0
		else:
			self.index = self.layer_prev.index+1

		weight_index = 0
		weights_per_neuron = input_count + 1


		for i in range(0, neuron_count):
			n = Neuron(i, weights[weight_index:weight_index+weights_per_neuron], self)
			weight_index+=weights_per_neuron
			self.neurons[i]=n

	def add_next_layer(self, neuron_count, weights=None):
		if self.layer_next!=None:
			return self.layer_next.add_next_layer(neuron_count, weights)

		w = weights
		if w==None:
			w=Perceptron.generate_weights(len(self.neurons), neuron_count)
		
		next = Perceptron(len(self.neurons), neuron_count, w, self)
		self.layer_next = next
		return next

	def evaluate_error(self, inputs, expected):
		results = self.evaluate(inputs)
		error = 0
		for i in range(0, len(expected)):
			if results[i]!=expected[i]:
				error+=1
		error=error/len(expected)

		return error

	def evaluate(self, inputs, dropout_rate=0.0):
		results = len(self.neurons)*[0]
		layer_dropout_rate=0.0

		# If this is a hidden layer, apply dropout (if set)
		if self.layer_prev!=None and self.layer_next!=None:
			layer_dropout_rate = dropout_rate

		for i in range(0, len(self.neurons)):
			if layer_dropout_rate>0:
				if random.random()<=layer_dropout_rate:
					self.neurons[i].dropout=True
					results[i]=0
					continue
			self.neurons[i].dropout = False
			results[i]=self.neurons[i].evaluate(inputs)
		if self.layer_next!=None:
			return self.layer_next.evaluate(results, dropout_rate)
		else:
			# If output layer, convert to boolean outputs
			for i in range(0, len(results)):
				results[i]=Neuron.to_output(results[i])
			return results
	# Calculate the error for this layer applicable to the 
	# specified neuron on the previous layer
	# En=Sum(Win*Ei)
	def get_error_for(self, previous_layer_neuron_index):
		output = 0
		# Step through each neuron on this layer
		for i in range(0, len(self.neurons)):
			#index+1 to account for bias (weight[0] is a bias, not fed by a neuron)
			if self.neurons[i].dropout!=True:
				output+=(self.neurons[i].weights[previous_layer_neuron_index+1]
							*self.neurons[i].error)
		return output

	def backpropagate(self, inputs, expected, learning_rate):
		if self.layer_prev is None:
			# Input layer
			self.evaluate(inputs) #, 0.50)
		e = 0
		if self.layer_next!=None:
			# Not output layer
			self.layer_next.backpropagate(None, expected, learning_rate)
			for i in range(0, len(self.neurons)):
				if self.neurons[i].dropout==True:
					continue
				err = self.layer_next.get_error_for(i)
				e = self.neurons[i].backpropagate(err)

		else:
			# Output layer
			total_errors = 0
			for i in range(0, len(self.neurons)):
				e = self.neurons[i].calculate_error(expected[i])
				self.neurons[i].backpropagate(e)
				lg = self.neurons[i].last_guessedb()
				if lg!=expected[i]:
					total_errors+=1
			# Adjust learning_rate
			learning_rate*=(1-total_errors/len(self.neurons))
			if learning_rate>1.0:
				learning_rate=1.0
			elif learning_rate<=0.0001:
				learning_rate=0.0001
		# Now update the weights
		if self.layer_prev==None:
			# Back to input layer
			self.backpropagate_update_weights(inputs, learning_rate)

		return e
	def backpropagate_update_weights(self, inputs, learning_rate):
		results = len(self.neurons)*[0]
		for i in range(0, len(self.neurons)):
			if self.neurons[i].dropout!=True:
				results[i]=self.neurons[i].backpropagate_update_weights(inputs, learning_rate)
			else:
				results[i]=0
		if self.layer_next!=None:
			self.layer_next.backpropagate_update_weights(results, learning_rate)
n=Neuron(1,[],3)

def test_AND():
	p=Perceptron.new_perceptron_random(2, 40)
	p.add_next_layer(40)
	p.add_next_layer(1)

	print("%s %s" %([0,0], p.evaluate([0,0])))
	print("%s %s" %([0,1], p.evaluate([0,1])))
	print("%s %s" %([1,0], p.evaluate([1,0])))
	print("%s %s" %([1,1], p.evaluate([1,1])))

	for i in range(0, 1000):
		p.backpropagate([0,0], [0], 0.1)
		p.backpropagate([0,1], [0], 0.1)
		p.backpropagate([1,0], [0], 0.1)
		p.backpropagate([1,1], [1], 0.1)

	print("%s %s" %([0,0], p.evaluate([0,0])))
	print("%s %s" %([0,1], p.evaluate([0,1])))
	print("%s %s" %([1,0], p.evaluate([1,0])))
	print("%s %s" %([1,1], p.evaluate([1,1])))

def test_XOR():
	p=Perceptron.new_perceptron_random(2, 50)
	p.add_next_layer(50)
	#p.add_next_layer(50)
	p.add_next_layer(1)

	print("%s %s" %([0,0], p.evaluate([0,0])))
	print("%s %s" %([0,1], p.evaluate([0,1])))
	print("%s %s" %([1,0], p.evaluate([1,0])))
	print("%s %s" %([1,1], p.evaluate([1,1])))

	for i in range(0, 4000):
		p.backpropagate([0,0], [0], 0.1)
		p.backpropagate([0,1], [1], 0.1)
		p.backpropagate([1,0], [1], 0.1)
		p.backpropagate([1,1], [0], 0.1)

	print("%s %s" %([0,0], p.evaluate([0,0])))
	print("%s %s" %([0,1], p.evaluate([0,1])))
	print("%s %s" %([1,0], p.evaluate([1,0])))
	print("%s %s" %([1,1], p.evaluate([1,1])))

test_AND()
test_XOR()

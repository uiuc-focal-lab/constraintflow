
class Value:

	def __init__(self):
		pass

class IntValue(Value):

	def __init__(self, v):
		self.v = v

class BoolValue(Value):

	def __init__(self, v):
		self.v = v

class FloatValue(Value):

	def __init__(self, v):
		self.v = v

class NeuronValue(Value):

	def __init__(self, name):
		self.name = name

class PolyExpValue(Value):

	#Coeffs have type float if they could be concretely evaluated, but they can be values such as argmax(prev, l)[bias]
	def __init__(self, neurons, coeffs, ops)
		self.neurons = neurons
		self.coeffs = coeffs
		self.ops = ops

class ZonoExpValue(Value):

	def __init__(self, coeffs, ops)
		self.coeffs = coeffs
		self.ops = ops

class Metadata(Value):

	def __init__(self, neuron: NeuronValue, metadata):
		self.neuron = neuron
		self.metadata = metadata

class ShapeElement(Value):

	def __init__(self, neuron: NeuronValue, element):
		self.neuron = neuron
		self.element = element





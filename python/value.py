
class Value:

	def __init__(self):
		pass

class IntValue(Value):

	def __init__(self, v):
		self.v = v

	def __eq__(self, obj):
		if(isinstance(obj, IntValue)):
			return self.v == obj.v
		else:
			return False

class BoolValue(Value):

	def __init__(self, v):
		self.v = v

	def __eq__(self, obj):
		if(isinstance(obj, BoolValue)):
			return self.v == obj.v
		else:
			return False

class FloatValue(Value):

	def __init__(self, v):
		self.v = v

	def __eq__(self, obj):
		if(isinstance(obj, FloatValue)):
			return self.v == obj.v
		else:
			return False

class NeuronValue(Value):

	def __init__(self, name):
		self.name = name

	def __eq__(self, obj):
		if(isinstance(obj, NeuronValue)):
			return self.name == obj.name
		else:
			return False

class PolyExpValue(Value):

	#Coeffs have type float if they could be concretely evaluated, but they can be values such as argmax(prev, l)[bias]
	def __init__(self, coeffs, const):
		self.const = const #const : tuple
		self.coeffs = coeffs #{neuron name -> coeff: tuple}

	def __eq__(self, obj):
		if(isinstance(obj, PolyExpValue)):
			return self.coeffs == obj.coeffs and self.const == obj.const
		else:
			return False

class ZonoExpValue(Value):

	def __init__(self, coeffs):
		self.coeffs = coeffs #List of (coeff, op)

	def __eq__(self, obj):
		if(isinstance(obj, ZonoExpValue)):
			return self.coeffs == obj.coeffs 
		else:
			return False


class Metadata(Value):

	def __init__(self, neuron: NeuronValue, metadata):
		self.neuron = neuron
		self.metadata = metadata

	def __eq__(self, obj):
		if(isinstance(obj, Metadata)):
			return self.neuron == obj.neuron and self.metadata == obj.metadata
		else:
			return False

class ShapeElement(Value):

	def __init__(self, neuron: NeuronValue, element):
		self.neuron = neuron
		self.element = element

	def __eq__(self, obj):
		if(isinstance(obj, ShapeElement)):
			return self.neuron == obj.neuron and self.element == obj.element
		else:
			return False




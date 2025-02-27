#A value is either a TerminalValue or a tuple (x, t)
#where x is a symbolic variable and t is it's type, "Float", "Neuron", etc.
#M is a dictionary from NonTerminalValues to TerminalValues
#C is a list of z3 symbolic constraints
#V is a dictionary from a symbolic variable to the Vertex class of the neuron it represents

from z3 import *

class Vertex:
	#Store the expression for the vertex polyhedral element when it is defined(say if we are using a map function)
	def __init__(self, name):
		self.symmap = {} #ex) "layer" -> (Int('X1'), "Int")
		self.name = Real(name) #Should be unique name


class PolyExpValue():

	#Coeffs have type float if they could be concretely evaluated, but they can be values such as argmax(prev, l)[bias]
	def __init__(self, coeffs, const):
		self.const = const #const : tuple
		self.coeffs = coeffs #{neuron name -> coeff: tuple}

class SymExpValue():

	def __init__(self, coeffs, const):
		self.const = const #const : tuple
		self.coeffs = coeffs #{noise variable -> coeff: tuple}
		
class TerminalValue:
	def __init__(self):
		pass

class NonTerminalValue:
	def __init__(self):
		pass

class ADD(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, ADD)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("ADD", self.left, self.right))

class SUB(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, SUB)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("SUB", self.left, self.right))

class MULT(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, MULT)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("MULT", self.left, self.right))

def get_multiplicands(val):
	if isinstance(val, MULT):
		return get_multiplicands(val.left) + get_multiplicands(val.right)
	return [val]

def get_type(val):
	if(isinstance(val, tuple)):
		return val[1]
	else:
		l = get_type(val.left)
		if isinstance(val, ADD) or isinstance(val, SUB) or isinstance(val, MULT) or isinstance(val, DIV):
			if l=='Neuron' or l=='PolyExp':
				return 'PolyExp'
			elif l=='Noise' or l=='SymExp':
				return 'SymExp'
			else:
				return get_type(val.right)
		if isinstance(val, IF):
			r = get_type(val.right)
			if l=='Float' and r=='Float':
				return 'Float'
			if l=='Int' and r=='Int':
				return 'Int'
			if l=='Bool' and r=='Bool':
				return 'Bool'
			if l=='SymExp' or r=='SymExp' or l=='Noise' or r=='Noise':
				return 'SymExp'
			if l=='PolyExp' or r=='PolyExp' or l=='Neuron' or r=='Neuron':
				return 'PolyExp'

def create_add(left, right):
	if isinstance(left, tuple) and isinstance(right, tuple):
		if get_type(left)=='Float' and get_type(right)=='Float':
			return (left[0]+right[0], 'Float')
	return ADD(left, right)

def create_sub(left, right):
	if isinstance(left, tuple) and isinstance(right, tuple):
		if get_type(left)=='Float' and get_type(right)=='Float':
			return (left[0]-right[0], 'Float')
	return SUB(left, right)

def create_div(left, right):
	if isinstance(left, tuple) and isinstance(right, tuple):
		if get_type(left)=='Float' and get_type(right)=='Float':
			return (left[0]/right[0], 'Float')
	return DIV(left, right)

def create_mult(left, right):
	# return MULT(left, right)
	if isinstance(left, tuple) and isinstance(right, tuple):
		if get_type(left)=='Float' and get_type(right)=='Float':
			return (left[0]*right[0], 'Float')
		return MULT(left, right)
	
	# left is base case
	elif (isinstance(left, tuple) or get_type(left)=='Float') and isinstance(right, ADD):
		lhs = create_mult(left, right.left)
		rhs = create_mult(left, right.right)
		return create_add(lhs, rhs)
	elif (isinstance(left, tuple) or get_type(left)=='Float') and isinstance(right, SUB):
		return create_sub(create_mult(left, right.left), create_mult(left, right.right))
	elif (isinstance(left, tuple) or get_type(left)=='Float') and isinstance(right, IF):
		return IF(right.cond, create_mult(left, right.left), create_mult(left, right.right))
	elif (isinstance(left, tuple) or get_type(left)=='Float') and isinstance(right, MULT):
		multiplicands = [left] + get_multiplicands(right)
		coeff = 1
		others = []
		for i in range(len(multiplicands)):
			if isinstance(multiplicands[i], tuple):
				if multiplicands[i][1] == 'Float':
					coeff = coeff * multiplicands[i][0]
					# coeff = simplify(coeff)
				else:
					others.append(multiplicands[i])
			else:
				others.append(multiplicands[i])
		if len(others)==0:
			return (coeff, 'Float')
		if len(others)==1:
			return MULT((coeff, 'Float'), others[0])
		else:
			coeff = (coeff, 'Float')
			for i in range(len(others)):
				coeff = MULT(coeff, others[i])
			return coeff 
		
	# right is base case
	elif isinstance(left, ADD) and (isinstance(right, tuple) or get_type(right) == 'Float'):
		return create_add(create_mult(left.left, right), create_mult(left.right, right))
	elif isinstance(left, SUB) and (isinstance(right, tuple) or get_type(right) == 'Float'):
		return create_sub(create_mult(left.left, right), create_mult(left.right, right))
	elif isinstance(left, IF) and (isinstance(right, tuple) or get_type(right) == 'Float'):
		return IF(left.cond, create_mult(left.left, right), create_mult(left.right, right))
	elif isinstance(left, MULT) and (isinstance(right, tuple) or get_type(right) == 'Float'):
		return create_mult(right, left)
	

	else:
		return MULT(left, right)
	

class DIV(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, DIV)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("DIV", self.left, self.right))

class LEQ(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, LEQ)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("LEQ", self.left, self.right))

class LT(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, LT)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("LT", self.left, self.right))

class GEQ(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, GEQ)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("GEQ", self.left, self.right))

class GT(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, GT)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("GT", self.left, self.right))

class EQQ(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, EQQ)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("EQQ", self.left, self.right))
	
class IN(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, IN)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("IN", self.left, self.right))

class NEQ(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, NEQ)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("NEQ", self.left, self.right))

class AND(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, AND)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("AND", self.left, self.right))

class OR(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, OR)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("OR", self.left, self.right))

class NOT(TerminalValue):

	def __init__(self, right):
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, NOT)):
			return self.right == obj.right

	def __hash__(self):
		return hash(("NOT", self.right))

class NEG(TerminalValue):

	def __init__(self, right):
		self.right = right
		self.left = right

	def __eq__(self, obj):
		if(isinstance(obj, NEG)):
			return self.right == obj.right

	def __hash__(self):
		return hash(("NEG", self.right))

class IF(TerminalValue):

	def __init__(self, cond, left, right):
		self.cond = cond 
		self.left = left 
		self.right = right 

	def __eq__(self, obj):
		if(isinstance(obj, IF)):
			return (self.cond==obj.cond and self.left==obj.left and self.right==obj.right)

	def __hash__(self):
		return hash(('IF', self.cond, self.left, self.right))

class MAX(NonTerminalValue):

	def __init__(self, e):
		self.e = e

	def __eq__(self, obj):
		if(isinstance(obj, MAX)):
			return self.e == obj.e
		else:
			return False

	def __hash__(self):
		return hash(("MAX", str(self.e)))

class MIN(NonTerminalValue):

	def __init__(self, e):
		self.e = e

	def __eq__(self, obj):
		if(isinstance(obj, MIN)):
			return self.e == obj.e
		else:
			return False

	def __hash__(self):
		return hash(("MIN", str(self.e)))

class TRAVERSE(NonTerminalValue):

	def __init__(self, e, d, f1, f2, f3):
		self.e = e
		self.d = d
		self.f1 = f1
		self.f2 = f2
		self.f3 = f3

	def __eq__(self, obj):
		if(isinstance(obj, TRAVERSE)):
			return self.e == obj.e and self.d == obj.d and self.f1 == obj.f1 and self.f2 == obj.f2 and self.f3 == obj.f3
		else:
			return False

	def __hash__(self):
		return 0
		return hash(("TRAVERSE", str(self.e), str(self.d), str(self.f1), str(self.f2), str(self.f3)))

class LP(NonTerminalValue):

	def __init__(self, op, e, c):
		self.op = op
		self.e = e
		self.c = c

	def __eq__(self, obj):
		if(isinstance(obj, LP)):
			return self.op == self.op and self.e == obj.e and self.c == obj.c
		else:
			return False

	def __hash__(self):
		return hash(("LP", self.op, self.e, str(self.c)))

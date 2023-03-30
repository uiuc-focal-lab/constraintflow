#Output of Evaluate.visit() functions is a value
#A value is either a TerminalValue or a tuple (x, t)
#where x is a symbolic variable and t is it's type, "Float", "Neuron", etc.
#M is a dictionary from NonTerminalValues to TerminalValues
#C is a list of z3 symbolic constraints
#V is a dictionary from a symbolic variable to the Vertex class of the neuron it represents



class TerminalValue:
	def __init__(self):
		pass

class NonTerminalValue:
	def __init__(self):
		pass

class Add(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, Add)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("Add", self.left, self.right))

class Sub(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, Sub)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("Sub", self.left, self.right))

class Mult(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, Mult)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("Mult", self.left, self.right))

class Div(TerminalValue):

	def __init__(self, left, right):
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, Div)):
			return self.left == obj.left and self.right == obj.right

	def __hash__(self):
		return hash(("Div", self.left, self.right))

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

	def __eq__(self, obj):
		if(isinstance(obj, NEG)):
			return self.right == obj.right

	def __hash__(self):
		return hash(("NEG", self.right))


class Ternary(NonTerminalValue):

	def __init__(self, cond, left, right):
		self.cond = cond
		self.left = left
		self.right = right

	def __eq__(self, obj):
		if(isinstance(obj, Ternary)):
			return self.cond == obj.cond and self.left == obj.left and self.right == obj.right
		else:
			return False

	def __hash__(self):
		return hash(("Ternary", self.cond, self.left, self.right))

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

class ARGMAX(NonTerminalValue):

	def __init__(self, e, s):
		self.e = e
		self.s = s

	def __eq__(self, obj):
		if(isinstance(obj, ARGMAX)):
			return self.e == obj.e and self.s == obj.s
		else:
			return False

	def __hash__(self):
		return hash(("ARGMAX", str(self.e), self.s))

class ARGMIN(NonTerminalValue):

	def __init__(self, e, s):
		self.e = e
		self.s = s

	def __eq__(self, obj):
		if(isinstance(obj, ARGMIN)):
			return self.e == obj.e and self.s == obj.s
		else:
			return False

	def __hash__(self):
		return hash(("ARGMIN", str(self.e), self.s))

class LISTSUB(NonTerminalValue):

	def __init__(self, l, e):
		self.l = l
		self.s = s

	def __eq__(self, obj):
		if(isinstance(obj, LISTSUB)):
			return self.l == obj.l and self.s == obj.s
		else:
			return False

class Traverse(NonTerminalValue):

	def __init__(self, e, d, f1, f2, f3):
		self.e = e
		self.d = d
		self.f1 = f1
		self.f2 = f2
		self.f3 = f3

	def __eq__(self, obj):
		if(isinstance(obj, Traverse)):
			return self.e == obj.e and self.d == obj.d and self.f1 == obj.f1 and self.f2 == obj.f2 and self.f3 == obj.f3
		else:
			return False

	def __hash__(self):
		return hash(("Traverse", self.e, self.d, self.f1, self.f2, self.f3))
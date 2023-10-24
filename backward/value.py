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

# class LIST(TerminalValue):
# 	def __init__(self, l, f):
# 		self.elist = l 
# 		self.list_func = f 

# 	def __eq__(self, obj):
# 		if(isinstance(obj, LIST)):
# 			return self.list_func.name() == obj.list_func.name() and self.elist == obj.elist 
	
# 	def __hash__(self):
# 		return (('LIST', str(self.elist), self.list_func.name()))


# class EPSILON(NonTerminalValue):

# 	def __init__(self, identifier):
# 		self.identifier = identifier

# 	def __eq__(self, obj):
# 		if(isinstance(obj, EPSILON)):
# 			return self.identifier == obj.identifier
# 		else:
# 			return False

# 	def __hash__(self):
# 		return hash(("EPSILON", self.identifier))

# class TERNARY(NonTerminalValue):

# 	def __init__(self, cond, left, right):
# 		self.cond = cond
# 		self.left = left
# 		self.right = right

# 	def __eq__(self, obj):
# 		if(isinstance(obj, TERNARY)):
# 			return self.cond == obj.cond and self.left == obj.left and self.right == obj.right
# 		else:
# 			return False

# 	def __hash__(self):
# 		return hash(("TERNARY", self.cond, self.left, self.right))

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

# class ARGMAX(NonTerminalValue):

# 	def __init__(self, e, f, arglist = []):
# 		self.e = e
# 		self.f = f
# 		self.arglist = arglist

# 	def __eq__(self, obj):
# 		if(isinstance(obj, ARGMAX)):
# 			return self.e == obj.e and self.f == obj.f and self.arglist == obj.arglist 
# 		else:
# 			return False

# 	def __hash__(self):
# 		return hash(("ARGMAX", str(self.e), self.f))

# class ARGMIN(NonTerminalValue):

# 	def __init__(self, e, s):
# 		self.e = e
# 		self.s = s

# 	def __eq__(self, obj):
# 		if(isinstance(obj, ARGMIN)):
# 			return self.e == obj.e and self.s == obj.s
# 		else:
# 			return False

# 	def __hash__(self):
# 		return hash(("ARGMIN", str(self.e), self.s))


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
		return hash(("TRAVERSE", self.e, self.d, self.f1, self.f2, self.f3))

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

# class DOT(NonTerminalValue):

# 	def __init__(self, l1, l2):
# 		self.list1 = l1 #LIST TerminalValue
# 		self.list2 = l2

# 	def __eq__(self, obj):
# 		if(isinstance(obj, LIST)):
# 			return self.list1 == obj.list1 and self.list2 == obj.list2
	
# 	def __hash__(self):
# 		return hash(("DOT", l1, l2))

# class GETELEMENT(NonTerminalValue):
# 	def __init__(self, elist, elem):
# 		self.elist = elist
# 		self.elem = elem 

# 	def __eq__(self, obj):
# 		if(isinstance(obj, GETELEMENT)):
# 			return self.elist==obj.elist and self.elem==obj.elem 

# 	def __hash__(self):
# 		return hash(('GETELEMENT', elist, elem))

# class GETMETADATA(NonTerminalValue):
# 	def __init__(self, elist, metadata):
# 		self.elist = elist
# 		self.metadata = metadata 

# 	def __eq__(self, obj):
# 		if(isinstance(obj, GETMETADATA)):
# 			return self.elist==obj.elist and self.metadata==obj.metadata 

# 	def __hash__(self):
# 		return hash(('GETMETADATA', elist, metadata))
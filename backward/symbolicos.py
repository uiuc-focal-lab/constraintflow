import astVisitor
import ast as AST
from z3 import *
from value import *
import astPrinter
import copy
import itertools

class Number:

	def __init__(self):
		self.i = 0

	def nextn(self):
		self.i = self.i + 1
		return self.i

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

	# def __eq__(self, obj):
	# 	if(isinstance(obj, PolyExpValue)):
	# 		return self.coeffs == obj.coeffs and self.const == obj.const
	# 	else:
	# 		return False

class ZonoExpValue():

	def __init__(self, coeffs, const):
		self.const = const #const : tuple
		self.coeffs = coeffs #{noise variable -> coeff: tuple}

	# def __eq__(self, obj):
	# 	if(isinstance(obj, ZonoExpValue)):
	# 		return self.coeffs == obj.coeffs 
	# 	else:
	# 		return False


#Most of the visit functions return a value. 
#The property visit functions return a symbolic constraint
class SymbolicOperationalSemantics(astVisitor.ASTVisitor):

	def __init__(self, store, F, M, V, C, E, old_eps, old_neurons, shape, Nprev, Nzono):
		self.F = F #{function name -> function info}
		self.M = M #{(op, value) -> value}
		self.V = V #{symbolic var -> vertex}
		self.C = C #symbolic constraints
		self.E = E 
		self.old_eps = old_eps 
		self.old_neurons = old_neurons 
		self.store = store #{var -> value}
		self.shape = shape #{name -> type} of each variable in the shape
		self.vname = 0
		self.limit = 3 #max 3 neurons are connected to each one
		self.number = Number()
		self.symb = True
		self.Nprev = Nprev
		self.Nzono = Nzono

	def getVname(self):
		self.vname = self.vname + 1
		return "V" + str(self.vname)

	'''
	def visitBaseType(self, node: AST.BaseTypeNode):
	

	def visitArrayType(self, node: AST.ArrayTypeNode):
	

	def visitArgList(self, node: AST.ArgListNode):
	'''

	
		
	def ConstToPoly(self, const):
		return PolyExpValue({}, const)

	def NeuronToPoly(self, n):
		return PolyExpValue({n: (1, "Float")}, (0, "Float"))

	def ConstToZono(self, const):
		return ZonoExpValue({}, const)

	def NoiseToZono(self, n):
		return ZonoExpValue({n: (1, "Float")}, (0, "Float"))

	def ADDPoly(self, left, right):
		if(isinstance(left, tuple)):
			if(left[1] == "Neuron"):
				left = self.NoiseToZono(left)
			else:
				left = self.ConstToZono(left)

		if(isinstance(right, tuple)):
			if(right[1] == "Neuron"):
				right = self.NoiseToZono(right)
			else:
				right = self.ConstToZono(right)

		c = ADD(left.const, right.const)
		if isinstance(left.const, tuple):
			if left.const[0]==0:
				c = right.const 
		if isinstance(right.const, tuple):
			if right.const[0]==0:
				c = left.const 
		# (left.const[0] + right.const[0], "Float")
		n = {}
		for leftn in left.coeffs.keys():
			if(leftn in right.coeffs.keys()):
				n[leftn] = ADD(left.coeffs[leftn], right.coeffs[leftn])
				# n[leftn] = (left.coeffs[leftn][0] + right.coeffs[leftn][0], "Float")
			else:
				n[leftn] = left.coeffs[leftn]

		for rightn in right.coeffs.keys():
			if(not rightn in n.keys()):
				n[rightn] = right.coeffs[rightn]
				# n[rightn] = (right.coeffs[rightn][0], "Float")

		return ZonoExpValue(n, c)


	def ADDZono(self, left, right):
		if(isinstance(left, tuple)):
			if(left[1] == "Noise"):
				left = self.NoiseToZono(left)
			else:
				left = self.ConstToZono(left)

		if(isinstance(right, tuple)):
			if(right[1] == "Noise"):
				right = self.NoiseToZono(right)
			else:
				right = self.ConstToZono(right)

		c = ADD(left.const, right.const)
		if isinstance(left.const, tuple):
			if left.const[0]==0:
				c = right.const 
		if isinstance(right.const, tuple):
			if right.const[0]==0:
				c = left.const 
		# (left.const[0] + right.const[0], "Float")
		n = {}
		for leftn in left.coeffs.keys():
			if(leftn in right.coeffs.keys()):
				n[leftn] = ADD(left.coeffs[leftn], right.coeffs[leftn])
				# n[leftn] = (left.coeffs[leftn][0] + right.coeffs[leftn][0], "Float")
			else:
				n[leftn] = left.coeffs[leftn]

		for rightn in right.coeffs.keys():
			if(not rightn in n.keys()):
				n[rightn] = right.coeffs[rightn]
				# n[rightn] = (right.coeffs[rightn][0], "Float")

		return ZonoExpValue(n, c)


	def SUBPoly(self, left, right):
		if(isinstance(left, tuple)):
			if(left[1] == "Neuron"):
				left = self.NeuronToPoly(left)
			else:
				left = self.ConstToPoly(left)

		if(isinstance(right, tuple)):
			if(right[1] == "Neuron"):
				right = self.NeuronToPoly(right)
			else:
				right = self.ConstToPoly(right)

		c = SUB(left.const, right.const)
		
		n = {}
		for leftn in left.coeffs.keys():
			if(leftn in right.coeffs.keys()):
				n[leftn] = SUB(left.coeffs[leftn], right.coeffs[leftn])
				# n[leftn] = (left.coeffs[leftn][0] - right.coeffs[leftn][0], "Float")
			else:
				n[leftn] = left.coeffs[leftn]

		for rightn in right.coeffs.keys():
			if(not rightn in n.keys()):
				n[rightn] = SUB((0, 'Float'), right.coeffs[rightn])

		return PolyExpValue(n, c)

	def SUBZono(self, left, right):
		if(isinstance(left, tuple)):
			if(left[1] == "Noise"):
				left = self.NoiseToZono(left)
			else:
				left = self.ConstToZono(left)

		if(isinstance(right, tuple)):
			if(right[1] == "Noise"):
				right = self.NoiseToZono(right)
			else:
				right = self.ConstToZono(right)

		c = SUB(left.const, right.const)
		if isinstance(left.const, tuple):
			if left.const[0]==0:
				c = MULT(right.const, (-1, 'Float')) 
		if isinstance(right.const, tuple):
			if right.const[0]==0:
				c = left.const 
		n = {}
		for leftn in left.coeffs.keys():
			if(leftn in right.coeffs.keys()):
				n[leftn] = SUB(left.coeffs[leftn], right.coeffs[leftn])
			else:
				n[leftn] = left.coeffs[leftn]

		for rightn in right.coeffs.keys():
			if(not rightn in n.keys()):
				n[rightn] = SUB((0, 'Float'), right.coeffs[rightn])

		return ZonoExpValue(n, c)

	def easy_mult(self, l, r):
		if isinstance(l, tuple):
			if l[0]==0:
				return l 
			if l[0]==1:
				return r 
		if isinstance(r, tuple):
			if r[0]==0:
				return r 
			if r[0]==1:
				return l 
		return MULT(l, r)

	def cond_mult(self, c, e):
		e.const  = MULT(e.const, IF(c, 1, 0))
		for i in e.coeffs:
			e.coeffs[i] = MULT(e.coeffs[i], IF(c, 1, 0))
		return e

	def MULTZono(self, left, right):
		if len(left.coeffs)==0:
			right.const = self.easy_mult(right.const, left.const)
			for i in right.coeffs:
				right.coeffs[i] = self.easy_mult(right.coeffs[i], left.const)
			return right 
		else:
			left.const = self.easy_mult(left.const, right.const)
			for i in left.coeffs:
				left.coeffs[i] = self.easy_mult(left.coeffs[i], right.const)
			return left 

	def DIVZono(self, left, right):
		left.const = DIV(left.const, right.const)
		for i in left.coeffs:
			if isinstance(right.const, tuple):
				if not right.const[0]==1:
					left.coeffs[i] = DIV(left.coeffs[i], right.const)
			else:
				left.coeffs[i] = DIV(left.coeffs[i], right.const)
		return left 

	def MULTPoly(self, left, right):
		if len(left.coeffs)==0:
			right.const = self.easy_mult(right.const, left.const)
			for i in right.coeffs:
				right.coeffs[i] = self.easy_mult(right.coeffs[i], left.const)
			return right 
		else:
			left.const = self.easy_mult(left.const, right.const)
			for i in left.coeffs:
				left.coeffs[i] = self.easy_mult(left.coeffs[i], right.const)
			return left 

	def DIVPoly(self, left, right):
		left.const = DIV(left.const, right.const)
		for i in left.coeffs:
			if isinstance(right.const, tuple):
				if not right.const[0]==1:
					left.coeffs[i] = DIV(left.coeffs[i], right.const)
			else:
				left.coeffs[i] = DIV(left.coeffs[i], right.const)
		return left

	def convertToPoly(self, node):
		if(isinstance(node, tuple)):
			if(node[1] == "Float" or node[1] == "Int"):
				return PolyExpValue({}, node)
			elif(node[1] == "Neuron"):
				return PolyExpValue({node: (1, "Float")}, (0, "Float"))
		elif(isinstance(node, ADD)):
			left = self.convertToPoly(node.left)
			right = self.convertToPoly(node.right)
			return self.ADDPoly(left, right)
		elif(isinstance(node, SUB)):
			left = self.convertToPoly(node.left)
			right = self.convertToPoly(node.right)
			return self.SUBPoly(left, right)
		elif(isinstance(node, MULT)):
			left = self.convertToPoly(node.left)
			right = self.convertToPoly(node.right)
			return self.MULTPoly(left, right)
		elif(isinstance(node, DIV)):
			left = self.convertToPoly(node.left)
			right = self.convertToPoly(node.right)
			return self.DIVPoly(left, right)
		elif(isinstance(node, IF)):
			if self.get_type(node)=='Float':
				return PolyExpValue({}, node)
			elif self.get_type(node) == 'Noise':
				return PolyExpValue({node : (1, 'Float')}, (0, 'Float'))
			else:
				left = self.convertToPoly(node.left)
				right = self.convertToPoly(node.right)
				return ADDPoly(self.cond_mult(node.cond, left), self.cond_mult(NOT(node.cond), right))
		else:
			print(node)
			assert False

	

	def convertToZono(self, node):
		if(isinstance(node, tuple)):
			if(node[1] == "Float" or node[1] == "Int"):
				return ZonoExpValue({}, node)
			elif(node[1] == "Noise"):
				return ZonoExpValue({node: (1, "Float")}, (0, "Float"))
		elif(isinstance(node, ADD)):
			left = self.convertToZono(node.left)
			right = self.convertToZono(node.right)
			return self.ADDZono(left, right)
		elif(isinstance(node, SUB)):
			left = self.convertToZono(node.left)
			right = self.convertToZono(node.right)
			return self.SUBZono(left, right)
		elif(isinstance(node, MULT)):
			left = self.convertToZono(node.left)
			right = self.convertToZono(node.right)
			return self.MULTZono(left, right)
		elif(isinstance(node, DIV)):
			left = self.convertToZono(node.left)
			right = self.convertToZono(node.right)
			return self.DIVZono(left, right)
		elif(isinstance(node, IF)):
			if self.get_type(node)=='Float':
				return ZonoExpValue({}, node)
			elif self.get_type(node) == 'Noise':
				return ZonoExp({node : (1, 'Float')}, (0, 'Float'))
			else:
				left = self.convertToZono(node.left)
				right = self.convertToZono(node.right)
				return ADDZono(self.cond_mult(node.cond, left), self.cond_mult(NOT(node.cond), right))

		# elif(isinstance(node, MULT)):
		# 	if(self.get_type(node.left) == "ZonoExp"):
		# 		return ZonoExpValue({node.left: self.convertToZono(node.right)}, (0, "Float"))
		# 	else:
		# 		return ZonoExpValue({node.right: self.convertToZono(node.left)}, (0, "Float"))
		# elif(isinstance(node, DIV)):
		# 	return ZonoExpValue({node.left: (1 / (self.convertToZono(node.right).const))}, (0, "Float"))
		else:
			print(node)
			assert False

	def convertToZ3(self, node):

		if(isinstance(node, tuple)):
			return node[0]
		if(isinstance(node, list)):
			if(len(node) == 0):
				return True
			else:
				return And(self.convertToZ3(node[0]),self.convertToZ3(node[1:]))
		elif(isinstance(node, ADD)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			if isinstance(l, float):
				if l==0:
					return r 
			if isinstance(r, float):
				if r==0:
					return l 
			return l + r
		elif(isinstance(node, SUB)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			if isinstance(l, float):
				if l==0:
					return -r 
			if isinstance(r, float):
				if r==0:
					return l 
			return l - r
		elif(isinstance(node, MULT)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			if isinstance(l, float):
				if l==0:
					return 0 
				if l==1:
					return r 
			if isinstance(r, float):
				if r==0:
					return 0 
				if r==1:
					return l
			return l * r
		elif(isinstance(node, DIV)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			return l / r
		elif(isinstance(node, NEG)):
			l = self.convertToZ3(node.left)
			return -l
		elif(isinstance(node, NOT)):
			l = self.convertToZ3(node.left)
			if isinstance(l, bool):
				if l:
					return False 
				return True
			return Not(l)
		elif(isinstance(node, LT)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			return l < r
		elif(isinstance(node, GT)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			return l > r
		elif(isinstance(node, LEQ)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			return l <= r
		elif(isinstance(node, GEQ)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			return l >= r
		elif(isinstance(node, EQQ)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			# print(l)
			# print(r)
			return l == r
		elif(isinstance(node, NEQ)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			return Not(l == r)
		elif(isinstance(node, AND)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			if isinstance(l, bool):
				if l:
					return r 
				else:
					return False 
			if isinstance(r, bool):
				if r:
					return l 
				else:
					return False
			return And(l, r)
		elif(isinstance(node, OR)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			if isinstance(l, bool):
				if l:
					return True  
				else:
					return r  
			if isinstance(r, bool):
				if r:
					return True  
				else:
					return l  
			return Or(l,r)
		elif(isinstance(node, IF)):
			y = Real('new_'+str(self.number.nextn()))
			c = self.convertToZ3(node.cond)
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			self.C.append(Or([And(c, y == l), And(Not(c), y == r)]))
			return y
			return If(c, l, r)
		else:
			return node

	def get_binop(self, left, right, f):
		return f(left, right)
		if isinstance(left, IF):
			return IF(left.cond, self.get_binop(left.left, right, f), self.get_binop(left.right, right, f))
		elif isinstance(right, IF):
			return IF(right.cond, self.get_binop(left, right.left, f), self.get_binop(left, right.right, f))
		else:
			return f(left, right)

	def get_unop(self, val, f):
		if isinstance(val, IF):
			return IF(val.cond, self.get_unop(val.left, f), self.get_unop(val.right, f))
		else:
			return f(val)

	def visitInt(self, node: AST.ConstIntNode):
		return (node.value, "Int")

	def visitFloat(self, node: AST.ConstFloatNode):
		return (node.value, "Float")
	
	def visitBool(self, node: AST.ConstBoolNode):
		return (node.value, "Bool")

	def visitVar(self, pt):
		return self.store[pt.name]

	def visitEpsilon(self, node: AST.EpsilonNode):
		eps = Real('Eps_'+str(self.number.nextn()))
		# self.C.append(eps <= 1)
		# self.C.append(eps >= -1)
		self.E.append(eps)
		return (eps, 'Noise')
		# return self.M[Epsilon(node.identifier)]
	
	def visitExprList(self, node: AST.ExprListNode):
		exps = []
		for exp in node.exprlist:
			exps.append(self.visit(exp))
		return exps

	def visitBinOp(self, node: AST.BinOpNode):
		left = self.visit(node.left)
		right = self.visit(node.right)
		if(node.op == "+"):
			return self.get_binop(left, right, ADD)
		elif(node.op == "-"):
			return self.get_binop(left, right, SUB)
		elif(node.op == "*"):
			return self.get_binop(left, right, MULT)
		elif(node.op == "/"):
			return self.get_binop(left, right, DIV)
		elif(node.op == "<="):
			return self.get_binop(left, right, LEQ)
		elif(node.op == "<"):
			return self.get_binop(left, right, LT)
		elif(node.op == ">="):
			return self.get_binop(left, right, GEQ)
		elif(node.op == ">"):
			return self.get_binop(left, right, GT)
		elif(node.op == "and"):
			return self.get_binop(left, right, AND)
		elif(node.op == "or"):
			return self.get_binop(left, right, OR)
		elif(node.op == "=="):
			return self.get_binop(left, right, EQQ)
		elif(node.op == "In"):
			return self.get_binop(left, right, EQQ)


	def visitUnOp(self, node: AST.UnOpNode):
		expr = self.visit(node.expr)
		# if isinstance(expr, IF):
		# 	return IF(expr.cond, self.visitUnOp(expr.left), self.visitUnOp(expr.right))
		# else:
		if(node.op == "-"):
			return self.get_unop(expr, NEG)
			# return NEG(expr)
		elif(node.op == "!"):
			return self.get_unop(expr, NOT)
			# return NOT(expr)

	def visitTernary(self, node: AST.TernaryNode):
		cond = self.visit(node.cond)
		left = self.visit(node.texpr)
		right = self.visit(node.fexpr)
		return IF(cond, left, right)
		# return self.M[TERNARY(cond, left, right)]

	
	def get_argmax(self, elist, node):
		if isinstance(elist, IF):
			return IF(elist.cond, self.get_argmax(elist.left, node), self.get_argmax(elist.right, node))
		else: 
			if isinstance(node.func, AST.VarNode):
				pre_elist = []
				fname = node.func
			else:	
				pre_elist = self.visit(node.func)
				fname = node.func.name

			n = len(elist)
			temp_array = []
			for i in range(n):
				temp = []
				for j in range(n):
					temp.append(None)
				temp_array.append(temp)

			for i in range(n):
				for j in range(n):
					if i == j:
						temp_array[i][j] = (True, 'Bool')
					else:
						if(node.op == "argmax"):
							arglist = AST.ExprListNode(pre_elist + [elist[i], elist[j]])
						else:
							arglist = AST.ExprListNode(pre_elist + [elist[j], elist[i]])
						fcall = AST.FuncCallNode(fname, arglist)
						temp_array[i][j] = self.visitFuncCall(fcall, True)
			combination = list(itertools.product([0, 1], repeat=n))
			out = []
			for k in range(1, len(combination)):
				c = combination[k]
				temp = []
				cond = (True, 'Bool')
				for i in range(n):
					if c[i]==1:
						for j in range(n):
							cond = self.get_binop(cond, temp_array[i][j], AND)
						temp.append(elist[i])
				out = IF(cond, temp, out)
			return out 

	def visitArgmaxOp(self, node: AST.ArgmaxOpNode):
		elist = self.visit(node.expr)
		return self.get_argmax(elist, node)

	def get_max(self, elist, node):
		if isinstance(elist, IF):
			return IF(elist.cond, self.get_max(elist.left, node), self.get_max(elist.right, node))
		else:
			n = len(elist)
			temp_array = []
			for i in range(n):
				temp = []
				for j in range(n):
					temp.append(None)
				temp_array.append(temp)
			for i in range(n):
				for j in range(n):
					if i == j:
						temp_array[i][j] = (True, 'Bool')
					else:
						if(node.op == "max"):
							temp_array[i][j] = self.get_binop(elist[i], elist[j], GEQ)
						else:
							temp_array[i][j] = self.get_binop(elist[i], elist[j], LEQ)
			out = elist[0]
			for i in range(1, n):
				cond = (True, 'Bool')
				for j in range(n):
					cond = self.get_binop(cond, temp_array[i][j], AND)
				out = IF(cond, elist[i], out)
			return out

	def visitMaxOpList(self, node: AST.MaxOpListNode):
		elist = self.visit(node.expr)
		return self.get_max(elist, node)


	def visitMaxOp(self, node: AST.MaxOpNode):
		e1 = self.visit(node.expr1)
		e2 = self.visit(node.expr2)
		if(node.op == "max"):
			return IF(self.get_binop(e1, e2, GEQ), e1, e2)
		else:
			return IF(self.get_binop(e1, e2, LEQ), e1, e2)

	def get_listOp(self, elist, node):
		if isinstance(elist, IF):
			return IF(elist.cond, self.get_listOp(elist.left, node), self.get_listOp(elist.right, node))
		else:
			sum = (0, "Float")
			length = (len(elist), 'Int')
			if len(elist)>0:
				sum = elist[0]
			for e in range(1, len(elist)):
				sum = self.get_binop(sum, elist[e], ADD)
			# for e in elist:
			# 	sum = self.get_binop(sum, e, ADD)
			if node.op == 'sum':
				return sum
			elif node.op == 'len':
				return length 
			elif node.op == 'avg':
				if(length[0] == 0):
					return 0
				return self.get_binop(sum, length, DIV)
			else:
				assert False

	def visitListOp(self, node: AST.ListOpNode):
		elist = self.visit(node.expr)
		return self.get_listOp(elist, node)

	def get_dot(self, left, right):
		if isinstance(left, IF):
			return IF(left.cond, self.get_dot(left.left, right), self.get_dot(left.right, right))
		elif isinstance(right, IF):
			return IF(right.cond, self.get_dot(left, right.left), self.get_dot(left, right.right))
		else:
			sum = (0, 'Float')
			n = min(len(left), len(right))
			for i in range(n):
				sum = self.get_binop(self.get_binop(left[i], right[i], MULT), sum, ADD)
			return sum
	
	def get_concat(self, left, right):
		if isinstance(left, IF):
			return IF(left.cond, self.get_dot(left.left, right), self.get_dot(left.right, right))
		elif isinstance(right, IF):
			return IF(right.cond, self.get_dot(left, right.left), self.get_dot(left, right.right))
		else:
			return left.append(right)

	def visitDot(self, node: AST.DotNode):
		left = self.visit(node.left)
		right = self.visit(node.right)
		return self.get_dot(left, right)

	def visitConcat(self, node: AST.ConcatNode):
		left = self.visit(node.left)
		right = self.visit(node.right)
		return self.get_concat(left, right)
	

	def get_getElement(self, expr, name):
		if isinstance(expr, IF):
			return IF(expr.cond, self.get_getElement(expr.left, name), self.get_getElement(expr.right, name))
		if isinstance(expr, list):
			out = []
			for e in expr:
				out.append(self.get_getElement(e, name))
			return out 
		else:
			return self.V[expr[0]].symmap[name]

	def visitGetElement(self, node):
		expr = self.visit(node.expr)
		return self.get_getElement(expr, node.elem.name)
		# if isinstance(elist, IF):
		# 	return IF(elist.cond, self.visitGetElement(elist.left, node.elem.name))
		# n = self.visit(node.expr)
		# if(isinstance(n, LIST)):
		# 	return self.M[GETELEMENT(n, node.elem.name)]
		# 	# return LIST([self.V[i[0]].symmap[node.elem.name] for i in n.elist])
		# else:
		# 	return self.V[n[0]].symmap[node.elem.name]

	def visitGetMetadata(self, node):
		expr = self.visit(node.expr)
		return self.get_getElement(expr, node.metadata.name)
		# n = self.visit(node.expr)
		# if(isinstance(n, list)):
		# 	return self.M[GETMETADATA(n, node.metadata.name)]
		# 	# return [self.V[i[0]].symmap[node.metadata.name] for i in n]
		# else:
		# 	return self.V[n[0]].symmap[node.metadata.name]

	def visitFuncCall(self, node: AST.FuncCallNode, preeval = False):
		if isinstance(node.name, str):
			func = self.F[node.name]
		else:
			func = self.F[node.name.name]
		newvars = []
		oldvalues = {}

		if(not preeval):
			elist = self.visit(node.arglist)
		else:
			elist = node.arglist.exprlist

		if len(elist)==len(func.decl.arglist.arglist):

			for (exp,(t, arg)) in zip(elist, func.decl.arglist.arglist):
				if arg.name in self.store.keys():
					oldvalues[arg.name] = self.store[arg.name]
				else:
					newvars.append(arg.name)

				self.store[arg.name] = exp

			val = self.visit(func.expr)
			
			for v in newvars:
				del self.store[v]

			for ov in oldvalues.keys():
				self.store[ov] = oldvalues[ov]

			#print(node.name.name)
			#print(val)
			return val

		elif len(elist)<=len(func.decl.arglist.arglist):
			return elist 
		else:
			assert False 

	def get_type(self, e):
		if isinstance(e, ADD) or isinstance(e, SUB) or isinstance(e, MULT) or isinstance(e, DIV):
			if self.get_type(e.left)=='Neuron' or self.get_type(e.left)=='PolyExp':
				return 'PolyExp'
			elif self.get_type(e.left)=='Noise' or self.get_type(e.left)=='ZonoExp':
				return 'ZonoExp'
			else:
				return self.get_type(e.right)
		else:
			if isinstance(e, IF):
				left = self.get_type(e.left)
				right = self.get_type(e.right)
				if left=='Float' and right=='Float':
					return 'Float'
				if left=='Int' and right=='Int':
					return 'Int'
				if left=='Bool' and right=='Bool':
					return 'Bool'
				if left=='ZonoExp' or right=='ZonoExp' or left=='Noise' or right=='Noise':
					return 'ZonoExp'
				if left=='PolyExp' or right=='PolyExp' or left=='Neuron' or right=='Neuron':
					return 'PolyExp'
			return e[1]


	def get_map(self, e, node):

		if isinstance(e, IF):
			return IF(e.cond, self.get_map(e.left, node), self.get_map(e.right, node))
		else:

			# if self.get_type(e) == 'PolyExp':
			# 	expr = self.convertToPoly(e)
			# else:
			# 	expr = self.convertToZono(e)
			if isinstance(e, ADD):
				l = self.get_map(e.left, node)
				r = self.get_map(e.right, node)
				# print(self.convertToZ3(l))
				# print(self.convertToZ3(e))
				return self.get_binop(self.get_map(e.left, node), self.get_map(e.right, node), ADD)
			elif isinstance(e, SUB):
				return self.get_binop(self.get_map(e.left, node), self.get_map(e.right, node), SUB)
			elif isinstance(e, MULT):
				lhstype = self.get_type(e.left)
				rhstype = self.get_type(e.right)

				if isinstance(node.func, AST.VarNode):
					elist = []
					fname = node.func
				else:	
					elist = self.visit(node.func)
					fname = node.func.name

				if(lhstype == 'PolyExp' or lhstype == 'ZonoExp' or lhstype == 'Neuron' or lhstype=='Noise'):
					elist = AST.ExprListNode(elist + [e.left, e.right])
					fcall = AST.FuncCallNode(node.func.name, elist)
					exp = self.visitFuncCall(fcall, True)
					# exp = self.get_binop(exp, self.visitFuncCall(fcall, True), ADD)
					return exp
				elif(rhstype == 'PolyExp' or rhstype == 'ZonoExp' or rhstype == 'Neuron' or rhstype=='Noise'):
					elist = AST.ExprListNode(elist + [e.right, e.left])
					fcall = AST.FuncCallNode(fname, elist)
					exp = self.visitFuncCall(fcall, True)
					fcall = AST.FuncCallNode(node.func.name, elist)
					exp = self.visitFuncCall(fcall, True)
					# exp = self.get_binop(exp, self.visitFuncCall(fcall, True), ADD)
					return exp
				else:
					return e
			elif isinstance(e, DIV):
				lhstype = self.get_type(e.left)
				if isinstance(node.func, AST.VarNode):
					elist = []
					fname = node.func
				else:	
					elist = self.visit(node.func)
					fname = node.funcname

				if(lhstype == 'PolyExp' or lhstype == 'ZonoExp' or lhstype == 'Neuron' or lhstype=='Noise'):
					elist = AST.ExprListNode(elist + [e.left, DIV(1,e.right)])
					fcall = AST.FuncCallNode(fname, elist)
					exp = self.get_binop(exp, self.visitFuncCall(fcall, True), ADD)
					return exp
				else:
					return e
			else: #tuple
				if(e[1] == 'Neuron' or e[1] == 'Noise'):
					if isinstance(node.func, AST.VarNode):
						elist = []
						fname = node.func
					else:	
						elist = self.visit(node.func)
						fname = node.func.name

					elist = AST.ExprListNode(elist + [e, 1])
					fcall = AST.FuncCallNode(fname, elist)

					exp = self.get_binop(exp, self.visitFuncCall(fcall, True), ADD)
					return exp
				else:
					return e


			# exp = expr.const
			# for n in expr.coeffs.keys():

			# 	if isinstance(node.func, AST.VarNode):
			# 		elist = []
			# 	else:	
			# 		elist = self.visit(node.func)
			# 	elist = AST.ExprListNode(elist + [n, expr.coeffs[n]])
			# 	fcall = AST.FuncCallNode(node.func.name, elist)
			# 	exp = self.get_binop(exp, self.visitFuncCall(fcall, True), ADD)
			# return exp

	# def get_map(self, e, node):
	# 	if isinstance(e, IF):
	# 		return IF(e.cond, self.get_map(e.left, node), self.get_map(e.right, node))
	# 	else:
	# 		if self.get_type(e) == 'PolyExp':
	# 			expr = self.convertToPoly(e)
	# 		else:
	# 			expr = self.convertToZono(e)
	# 		exp = expr.const
	# 		for n in expr.coeffs.keys():

	# 			if isinstance(node.func, AST.VarNode):
	# 				elist = []
	# 			else:	
	# 				elist = self.visit(node.func)
	# 			elist = AST.ExprListNode(elist + [n, expr.coeffs[n]])
	# 			fcall = AST.FuncCallNode(node.func.name, elist)
	# 			exp = self.get_binop(exp, self.visitFuncCall(fcall, True), ADD)
	# 		return exp

	def visitMap(self, node: AST.MapNode):
		e = self.visit(node.expr)
		# print(self.convertToZ3(e))
		return self.get_map(e, node)

	def get_get_mapList(self, e, node):
		if isinstance(e, IF):
			return IF(e.cond, self.get_get_mapList(e.left, node), self.get_get_mapList(self.right, node))
		else:
			if isinstance(node.func, AST.VarNode):
				elist = []
				fname = node.func
			else:	
				elist = self.visit(node.func)
				fname = node.func.name
			elist = AST.ExprListNode(elist + [e])
			fcall = AST.FuncCallNode(fname, elist)
			return self.visitFuncCall(fcall, True)

	def get_mapList(self, e, node):
		if isinstance(e, IF):
			return IF(e.cond, self.get_mapList(e.left, node), self.get_mapList(e.right, node))
		else:
			out = []
			for exp in e:
				out += [self.get_get_mapList(exp, node)]
			return out 

	def visitMapList(self, node: AST.MapNode):
		e = self.visit(node.expr)
		return self.get_mapList(e, node)

	def visitTraverse(self, node: AST.TraverseNode):
		e = self.visit(node.expr)

		if(isinstance(node.priority, AST.VarNode)):
			p_name = node.priority.name
		else:
			p_name = self.visit(node.priority)

		if(isinstance(node.stop, AST.VarNode)):
			s_name = node.stop.name
		else:
			s_name = self.visit(node.stop)
		return self.M[TRAVERSE(e, node.direction, p_name, s_name, node.func.name)]

	def visitLp(self, node):
		e = self.visit(node.expr)
		c = self.visit(node.constraints)
		return self.M[LP(node.op, e, c)]

	# def visitPropTermOp(self, prop):
	# 	left = self.visit(prop.leftpt)
	# 	right = self.visit(prop.rightpt)
	# 	if(prop.op == "+"):
	# 		return self.get_binop(left, right, ADD)
	# 	elif(prop.op == "-"):
	# 		return self.get_binop(left, right, SUB)

	# def visitDoubleProp(self, prop):
	# 	left = self.visit(prop.leftprop)
	# 	right = self.visit(prop.rightprop)
	# 	if(prop.op == "and"):
	# 		return self.get_binop(left, right, AND)
	# 	if(prop.op == "or"):
	# 		return self.get_binop(left, right, OR)

	# def visitPropTermBasic(self, prop):
	# 	return self.visit(prop.term)

	# def visitSingleProp(self, pt):
	# 	left = self.visit(pt.leftpt)
	# 	right = self.visit(pt.rightpt)
	# 	if(pt.op == "<"):
	# 		return self.get_binop(left, right, LT)
	# 	elif(pt.op == "<="):
	# 		return self.get_binop(left, right, LEQ)
	# 		# return left <= right
	# 	elif(pt.op == ">"):
	# 		return self.get_binop(left, right, GT)
	# 		# return left > right
	# 	elif(pt.op == ">="):
	# 		return self.get_binop(left, right, GEQ)
	# 		# return left >= right
	# 	elif(pt.op == "=="):
	# 		return self.get_binop(left, right, EQQ)
	# 		# return left == right
	# 	elif(pt.op == 'in'):
	# 		# right = self.convertToZono(right)
	# 		# max_right = right.const 
	# 		# min_right = right.const 
	# 		# zero = (0, 'Float')
	# 		# neg = (-1, 'Float')
	# 		# # sum = (0, 'Float')
	# 		# for c in right.coeffs:
	# 		# 	coeff = right.coeffs[c]
	# 		# 	abs = IF(self.get_binop(coeff, zero, GEQ), coeff, self.get_binop(coeff, neg, MULT))
	# 		# 	# sum = self.get_binop(sum, abs, ADD)
	# 		# 	max_right = self.get_binop(max_right, abs, ADD)
	# 		# 	min_right = self.get_binop(min_right, abs, SUB)
	# 		# # max_right = self.get_binop(max_right, (s, 'Float'), ADD)
	# 		# # min_right = self.get_binop(min_right, (s, 'Float'), SUB)
	# 		# leq = self.get_binop(left, max_right, LEQ)
	# 		# geq = self.get_binop(left, min_right, GEQ)
	# 		# return self.get_binop(leq, geq, AND)
	# 		return self.get_binop(left, right, EQQ)

	def visitFunc(self, node: AST.FuncNode):
		name = node.decl.name.name
		self.F[name] = node

	def visitSeq(self, node: AST.SeqNode):
		self.visit(node.stmt1)
		self.visit(node.stmt2)

	# def visitFlow(self, node: AST.FlowNode):
	# 	pass #verification should be done by this point

	# def visitProg(self, node: AST.ProgramNode):
	# 	self.visit(node.shape)
	# 	self.visit(node.stmt)

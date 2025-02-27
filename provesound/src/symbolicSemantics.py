from z3 import *
import itertools
import sys
import time

from constraintflow import astVisitor
from constraintflow import astcf as AST
from provesound.src.value import *
from provesound.lib.utils import *

sys.setrecursionlimit(5000) 



# The following class implements the symbolic semantics for ConstraintFlow.
# The semantics is implemented as a visitor for the abstract syntax tree.(ASTVisitor
# class from astConstraintFlow/astVisitor.py. The visitor for each node of the AST 
# returns a symbolic value defined in src/value.py. Further, in self.C, we store the 
# symbolic constraints generated as a result of the symbolic semantics as discussed 
# in the paper. This visitor implements the inference rules described in Section 5.3 
# of the paper and Appendix G.2.   
class SymbolicSemantics(astVisitor.ASTVisitor):

	def __init__(self, store, F, M, V, C, E, old_eps, old_neurons, shape, Nprev, Nsym, arrayLens):
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
		self.Nsym = Nsym
		self.flag = False
		self.tempC = []
		self.hasE = False
		self.arrayLens = arrayLens
		# self.arrayLens = dict()

	def getVname(self):
		self.vname = self.vname + 1
		return "V" + str(self.vname)
		
	def ConstToPoly(self, const):
		return PolyExpValue({}, const)

	def NeuronToPoly(self, n):
		return PolyExpValue({n: (1, "Float")}, (0, "Float"))

	def ConstToZono(self, const):
		return SymExpValue({}, const)

	def NoiseToZono(self, n):
		return SymExpValue({n: (1, "Float")}, (0, "Float"))

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

		c = create_add(left.const, right.const)
		if isinstance(left.const, tuple):
			if left.const[0]==0:
				c = right.const 
		if isinstance(right.const, tuple):
			if right.const[0]==0:
				c = left.const 
		n = {}
		for leftn in left.coeffs.keys():
			if(leftn in right.coeffs.keys()):
				n[leftn] = create_add(left.coeffs[leftn], right.coeffs[leftn])
			else:
				n[leftn] = left.coeffs[leftn]

		for rightn in right.coeffs.keys():
			if(not rightn in n.keys()):
				n[rightn] = right.coeffs[rightn]

		return PolyExpValue(n, c)


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

		c = create_add(left.const, right.const)
		if isinstance(left.const, tuple):
			if left.const[0]==0:
				c = right.const 
		if isinstance(right.const, tuple):
			if right.const[0]==0:
				c = left.const 
		n = {}
		for leftn in left.coeffs.keys():
			if(leftn in right.coeffs.keys()):
				n[leftn] = create_add(left.coeffs[leftn], right.coeffs[leftn])
			else:
				n[leftn] = left.coeffs[leftn]

		for rightn in right.coeffs.keys():
			if(not rightn in n.keys()):
				n[rightn] = right.coeffs[rightn]
				# n[rightn] = (right.coeffs[rightn][0], "Float")

		return SymExpValue(n, c)


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

		c = create_sub(left.const, right.const)
		
		n = {}
		for leftn in left.coeffs.keys():
			if(leftn in right.coeffs.keys()):
				n[leftn] = create_sub(left.coeffs[leftn], right.coeffs[leftn])
			else:
				n[leftn] = left.coeffs[leftn]

		for rightn in right.coeffs.keys():
			if(not rightn in n.keys()):
				n[rightn] = create_sub((0, 'Float'), right.coeffs[rightn])

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

		c = create_sub(left.const, right.const)
		if isinstance(left.const, tuple):
			if left.const[0]==0:
				c = create_mult(right.const, (-1, 'Float')) 
		if isinstance(right.const, tuple):
			if right.const[0]==0:
				c = left.const 
		n = {}
		for leftn in left.coeffs.keys():
			if(leftn in right.coeffs.keys()):
				n[leftn] = create_sub(left.coeffs[leftn], right.coeffs[leftn])
			else:
				n[leftn] = left.coeffs[leftn]

		for rightn in right.coeffs.keys():
			if(not rightn in n.keys()):
				n[rightn] = create_sub((0, 'Float'), right.coeffs[rightn])

		return SymExpValue(n, c)

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
		return create_mult(l, r)

	def cond_mult(self, c, e):
		e.const  = create_mult(e.const, IF(c, 1, 0))
		for i in e.coeffs:
			e.coeffs[i] = create_mult(e.coeffs[i], IF(c, 1, 0))
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
		left.const = create_div(left.const, right.const)
		for i in left.coeffs:
			if isinstance(right.const, tuple):
				if not right.const[0]==1:
					left.coeffs[i] = create_div(left.coeffs[i], right.const)
			else:
				left.coeffs[i] = create_div(left.coeffs[i], right.const)
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
		left.const = create_div(left.const, right.const)
		for i in left.coeffs:
			if isinstance(right.const, tuple):
				if not right.const[0]==1:
					left.coeffs[i] = create_div(left.coeffs[i], right.const)
			else:
				left.coeffs[i] = create_div(left.coeffs[i], right.const)
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
				temp1 = self.cond_mult(node.cond, left)
				temp2 = self.cond_mult(NOT(node.cond), right)
				ret = self.ADDPoly(temp1, temp2)
				return ret
		else:
			assert False

	

	def convertToZono(self, node):
		if(isinstance(node, tuple)):
			if(node[1] == "Float" or node[1] == "Int"):
				return SymExpValue({}, node)
			elif(node[1] == "Noise"):
				return SymExpValue({node: (1, "Float")}, (0, "Float"))
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
				return SymExpValue({}, node)
			elif self.get_type(node) == 'Noise':
				return SymExpValue({node : (1, 'Float')}, (0, 'Float'))
			else:
				left = self.convertToZono(node.left)
				right = self.convertToZono(node.right)
				return self.ADDZono(self.cond_mult(node.cond, left), self.cond_mult(NOT(node.cond), right))

		else:
			assert False
	
	def convertToZ3(self, node):

		if(isinstance(node, tuple)):
			return node[0]
		if(isinstance(node, list)):
			if(len(node) == 1):
				return self.convertToZ3(node[0])
			if(len(node) == 0):
				return True
			else:
				return And(self.convertToZ3(node[0]),self.convertToZ3(node[1:]))
		elif(isinstance(node, ADD)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			if isinstance(l, int) or isinstance(l, float):
				if l==0:
					return r 
			if isinstance(r, int) or isinstance(r, float):
				if r==0:
					return l 
			return l + r
		elif(isinstance(node, IN)):
			e1 = self.convertToZ3(node.left)
			e2 = self.convertToZ3(node.right)
			e2_max = get_z3max_eps(self.E, e2)
			e2_min = get_z3min_eps(self.E, e2)
			def remove(E, e):
				vars = z3_vars(e)
				for var in vars:
					if var in E:
						E.remove(var)
			remove(self.E, e2)
			if isinstance(e2_max, z3.z3.ArithRef):
				if e2.sexpr() == e2_max.sexpr():
					return e1 == e2 
			elif e2 == e2_max:
				return e1 == e2 
			self.tempC.append(one==1)
			self.tempC.append(minus_one==-1)
			return And(e1 <= e2_max, e1 >= e2_min)
		elif(isinstance(node, SUB)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			if isinstance(l, int) or isinstance(l, float):
				if l==0:
					return -r 
			if isinstance(r, int) or isinstance(r, float):
				if r==0:
					return l 
			return l - r
		elif(isinstance(node, MULT)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			if isinstance(l, int) or isinstance(l, float):
				if l==0:
					return 0 
				if l==1:
					return r 
			if isinstance(r, int) or isinstance(r, float):
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
			c = self.convertToZ3(node.cond)
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			return If(c, l, r)
		elif(isinstance(node, MAX)):
			e = [self.convertToZ3(i) for i in node.e]
			y = Real('new_'+str(self.number.nextn()))
			if(str(node.e) in self.arrayLens):
				lenexp = self.convertToZ3(self.arrayLens[str(node.e)])
				ge = []
				eq = []
				for i in range(len(e)):
					ge.append(Or(y>=e[i],(i+1)>lenexp))
					eq.append(And(y==e[i],(i+1)<=lenexp))
			else:
				ge = [(y>=i) for i in e]
				eq = [(y==i) for i in e]
			self.tempC.append(Or(eq))
			self.tempC.append(And(ge))
			return y
		elif(isinstance(node, MIN)):
			e = [self.convertToZ3(i) for i in node.e]
			y = Real('new_'+str(self.number.nextn()))
			if(str(node.e) in self.arrayLens):
				lenexp = self.convertToZ3(self.arrayLens[str(node.e)])
				le = []
				eq = []
				for i in range(len(e)):
					le.append(Or(y<=e[i],(i+1)>lenexp))
					eq.append(And(y==e[i],(i+1)<=lenexp))
			else:
				le = [(y<=i) for i in e]
				eq = [(y==i) for i in e]
			self.tempC.append(Or(eq))
			self.tempC.append(And(le))
			return y
		else:
			return node

	def get_binop(self, left, right, f):
		if isinstance(left, list):
			l = min(len(left), len(right))
			listout = [self.get_binop(left[i], right[i], f) for i in range(l)]
			xl = None
			xr = None
			if(str(left) in self.arrayLens):
				xl = self.arrayLens[str(left)]
			if(str(right) in self.arrayLens):
				xr = self.arrayLens[str(right)]
			
			if(xl == None and xr == None):
				return listout
			elif(xl == None):
				self.arrayLens[str(listout)] = xr
			elif(xr == None):
				self.arrayLens[str(listout)] = xl
			elif(isinstance(xl, tuple) and isinstance(xr, tuple) and (xl[0].sexpr() == xr[0].sexpr())):
				self.arrayLens[str(listout)] = xl
			else:
				self.arrayLens[str(listout)] = IF(LEQ(xr,xl), xr, xl)
			return listout

		if f==MULT:
			return create_mult(left, right)
		elif f==ADD:
			return create_add(left, right)
		elif f==SUB:
			return create_sub(left, right)
		elif f==DIV:
			return create_div(left, right)
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
		if 'prev' in pt.name:
			if isinstance(self.store[pt.name], list):
				return self.store[pt.name][:self.Nprev]
		return self.store[pt.name]

	def visitEpsilon(self, node: AST.EpsilonNode):
		self.hasE = True
		eps = Real('Eps_'+str(self.number.nextn()))
		self.C.append(eps <= 1)
		self.C.append(eps >= -1)
		self.E.append(eps)
		return (eps, 'Noise')
	
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
			return self.get_binop(left, right, IN)


	def visitUnOp(self, node: AST.UnOpNode):
		expr = self.visit(node.expr)
		if(node.op == "-"):
			return self.get_unop(expr, NEG)
		elif(node.op == "!"):
			return self.get_unop(expr, NOT)

	def visitTernary(self, node: AST.TernaryNode):
		cond = self.visit(node.cond)
		left = self.visit(node.texpr)
		right = self.visit(node.fexpr)
		return IF(cond, left, right)

	
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

			x = None
			if(str(elist) in self.arrayLens):
				x = self.arrayLens[str(elist)]

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
						if(x == None):
							temp_array[i][j] = self.visitFuncCall(fcall, True)
						elif(j > i):
							temp_array[i][j] = OR(GT(j+1,x),self.visitFuncCall(fcall, True))
						else:
							temp_array[i][j] = self.visitFuncCall(fcall, True)

			combination = list(itertools.product([0, 1], repeat=n))
			out = []
			for k in range(1, len(combination)):
				c = combination[k]
				temp = []
				cond = (True, 'Bool')
				for i in range(n):
					if c[i]==1:
						if(x != None):
							cond = AND(cond, LEQ(i+1,x))
						for j in range(n):
							cond = self.get_binop(cond, temp_array[i][j], AND)
						temp.append(elist[i])
				out = IF(cond, temp, out)
			return out 

	def visitArgmaxOp(self, node: AST.ArgmaxOpNode):
		elist = self.visit(node.expr)
		return self.get_argmax(elist, node)

	def get_max(self, elist, op):
		if isinstance(elist, IF):
			return IF(elist.cond, self.get_max(elist.left, op), self.get_max(elist.right, op))
		else:
			x = None
			if(str(elist) in self.arrayLens):
				x = self.arrayLens[str(elist)]

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
						if(op == "max"):
							temp_array[i][j] = self.get_binop(elist[i], elist[j], GEQ)
						else:
							temp_array[i][j] = self.get_binop(elist[i], elist[j], LEQ)
			out = elist[0]
			for i in range(1, n):
				if(x == None):
					cond = (True, "Bool")
				else:
					cond = LEQ(i+1,x)

				for j in range(n):
					if(x == None):
						cond = self.get_binop(cond, temp_array[i][j], AND)
					else:
						if(j > i):
							cond = self.get_binop(cond, OR(temp_array[i][j],GT(j+1,x)), AND)
						else:
							cond = self.get_binop(cond, temp_array[i][j], AND)

				out = IF(cond, elist[i], out)
			return out

	def visitMaxOpList(self, node: AST.MaxOpListNode):
		elist = self.visit(node.expr)
		if not isinstance(elist, list):
			raise Exception('This is not possible. Something must be wrong with type checking')
		
		if isinstance(elist[0], list):
			l = min([len(i) for i in elist])
			ret = []
			listlens = set()
			for listi in elist:
				if(str(listi) in self.arrayLens):
					listlens.add(self.arrayLens[str(listi)])

			for i in range(l):
				e = [j[i] for j in elist]
				ret.append(self.get_max(e, node.op))

			if(len(listlens) == 1):
				self.arrayLens[str(ret)] = listlens.pop()
			if(len(listlens) > 1):
				newlen = Int("newlen"+str(self.number.nextn()))
				for i in range(len(listlens)):
					self.tempC.append(newlen <= self.convertToZ3(listlens.pop()))
				self.arrayLens[str(ret)] = newlen
			return ret 

		ret =  self.get_max(elist, node.op)
		return ret

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
			if node.op in ['sum', 'len']:
				sum = (0, "Float")
				if(str(elist) in self.arrayLens):
					x = self.arrayLens[str(elist)]
					length = x
					
					if len(elist)>0:
						sum = IF(LEQ(1,x),elist[0],(0,"Int"))
					for e in range(1, len(elist)):
						sum = self.get_binop(sum, IF(LEQ(e+1,x),elist[e],(0,"Int")) , ADD)
				else:
					length = (len(elist), 'Int')

					if len(elist)>0:
						sum = elist[0]
					for e in range(1, len(elist)):
						sum = self.get_binop(sum, elist[e], ADD)
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
			elif node.op == 'avg':
				sum = (0, "Float")
				if(str(elist) in self.arrayLens):
					x = self.arrayLens[str(elist)]
					length = x
					
					if len(elist)>0:
						sum = IF(LEQ(1,x),self.get_binop(elist[0], length, DIV),(0,"Int"))
					for e in range(1, len(elist)):
						sum = self.get_binop(sum, IF(LEQ(e+1,x),self.get_binop(elist[e], length, DIV),(0,"Int")) , ADD)
				else:
					length = (len(elist), 'Int')

					if len(elist)>0:
						sum = self.get_binop(elist[0], length, DIV)
					for e in range(1, len(elist)):
						sum = self.get_binop(sum, self.get_binop(elist[e], length, DIV), ADD)
				if(length[0] == 0):
					return 0
				return sum
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
			xl = None
			xr = None
			if(str(left) in self.arrayLens):
				xl = self.arrayLens[str(left)]
			if(str(right) in self.arrayLens):
				xr = self.arrayLens[str(right)]

			n = min(len(left), len(right))
			sum = (0, 'Float')
			if(xl == None and xr == None):
				for i in range(n):
					sum = self.get_binop(sum, self.get_binop(left[i], right[i], MULT), ADD)
			elif(xl == None):
				for i in range(n):
					sum = self.get_binop(sum, IF(LEQ(i+1,xr),self.get_binop(left[i], right[i], MULT),(0,"Int")), ADD)
			elif(xr == None):
				for i in range(n):
					sum = self.get_binop(sum, IF(LEQ(i+1,xl),self.get_binop(left[i], right[i], MULT),(0,"Int")), ADD)
			elif(isinstance(xl, tuple) and isinstance(xr, tuple) and (xl[0].sexpr() == xr[0].sexpr())):
				for i in range(n):
					sum = self.get_binop(sum, IF(LEQ(i+1,xr),self.get_binop(left[i], right[i], MULT),(0,"Int")), ADD)
			else:
				for i in range(n):
					sum = self.get_binop(sum, IF(AND(LEQ(i+1,xr), LEQ(i+1,xl)),self.get_binop(left[i], right[i], MULT),(0,"Int")), ADD)

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
	

	def get_getElement(self, val, name):
		if isinstance(val, IF):
			return IF(val.cond, self.get_getElement(val.left, name), self.get_getElement(val.right, name))
		if isinstance(val, list):
			out = []
			for e in val:
				out.append(self.get_getElement(e, name))
			if(str(val) in self.arrayLens):
				self.arrayLens[str(out)] = self.arrayLens[str(val)]
			return out 
		else:
			return self.V[val[0]].symmap[name]

	def visitGetElement(self, node):
		val = self.visit(node.expr)
		return self.get_getElement(val, node.elem.name)

	def get_getElementAtIndex(self, expr, index):
		if isinstance(expr, IF):
			return IF(expr.cond, self.get_getElementAtIndex(expr.left, index), self.get_getElementAtIndex(expr.right, index))
		elif isinstance(expr, list):
			return expr[index] 
		else:
			raise Exception('Not possible')

	def visitGetElementAtIndex(self, node):
		expr = self.visit(node.expr)
		return self.get_getElementAtIndex(expr, node.index)
	
	def visitGetMetadata(self, node):
		expr = self.visit(node.expr)
		return self.get_getElement(expr, node.metadata.name)

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
				
			return val

		elif len(elist)<=len(func.decl.arglist.arglist):
			return elist 
		else:
			assert False 

	def get_type(self, e):
		if(isinstance(e, tuple)):
			return e[1]
		else:
			l = self.get_type(e.left)
			if isinstance(e, ADD) or isinstance(e, SUB) or isinstance(e, MULT) or isinstance(e, DIV):
				if l=='Neuron' or l=='PolyExp':
					return 'PolyExp'
				elif l=='Noise' or l=='SymExp':
					return 'SymExp'
				else:
					return self.get_type(e.right)
			if isinstance(e, IF):
				r = self.get_type(e.right)
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
		

	def get_map(self, e, node):
		if isinstance(e, IF):
			return IF(e.cond, self.get_map(e.left, node), self.get_map(e.right, node))
		else:
			if isinstance(e, ADD):
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

				if(lhstype == 'PolyExp' or lhstype == 'SymExp' or lhstype == 'Neuron' or lhstype=='Noise'):
					elist = AST.ExprListNode(elist + [e.left, e.right])
					fcall = AST.FuncCallNode(node.func.name, elist)
					exp = self.visitFuncCall(fcall, True)
					cond = self.get_binop(e.right, (0, "Float"), EQQ)
					
					
					exp_temp = IF(cond, (0, "Float"), exp)
					# OPTIMIZATION TO REMOVE IF
					if isinstance(fname, str):
						func = self.F[fname]
					else:
						func = self.F[fname.name]
					func_expr = func.expr
					func_args = func.decl.arglist.arglist
					for i in range(len(func_args)):
						if func_args[i][0].name == 'Float':
							coeff = func_args[i][1]
					if isinstance(func_expr, AST.BinOpNode) and func_expr.op=='*':
						if func_expr.left == coeff or func_expr.right==coeff:
							exp_temp = exp
					elif isinstance(func_expr, AST.TernaryNode):
						if isinstance(func_expr.left, AST.BinOpNode) and func_expr.left.op=='*':
							if func_expr.left.left == coeff or func_expr.left.right==coeff:
								if isinstance(func_expr.right, AST.BinOpNode) and func_expr.right.op=='*':
									if func_expr.right.left == coeff or func_expr.right.right==coeff:
										exp_temp = exp
					return exp_temp
				elif(rhstype == 'PolyExp' or rhstype == 'SymExp' or rhstype == 'Neuron' or rhstype=='Noise'):
					elist = AST.ExprListNode(elist + [e.right, e.left])
					fcall = AST.FuncCallNode(fname, elist)
					exp = self.visitFuncCall(fcall, True)
					fcall = AST.FuncCallNode(node.func.name, elist)
					exp = self.visitFuncCall(fcall, True)
					cond = self.get_binop(e.left, (0, "Float"), EQQ)


					exp_temp = IF(cond, (0, "Float"), exp)
					# OPTIMIZATION TO REMOVE IF
					if isinstance(fname, str):
						func = self.F[fname]
					else:
						func = self.F[fname.name]
					func_expr = func.expr
					func_args = func.decl.arglist.arglist
					for i in range(len(func_args)):
						if func_args[i][0].name == 'Float':
							coeff = func_args[i][1]
					if isinstance(func_expr, AST.BinOpNode) and func_expr.op=='*':
						if func_expr.left == coeff or func_expr.right==coeff:
							exp_temp = exp
					elif isinstance(func_expr, AST.TernaryNode):
						if isinstance(func_expr.left, AST.BinOpNode) and func_expr.left.op=='*':
							if func_expr.left.left == coeff or func_expr.left.right==coeff:
								if isinstance(func_expr.right, AST.BinOpNode) and func_expr.right.op=='*':
									if func_expr.right.left == coeff or func_expr.right.right==coeff:
										exp_temp = exp
					
					return exp_temp
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

				if(lhstype == 'PolyExp' or lhstype == 'SymExp' or lhstype == 'Neuron' or lhstype=='Noise'):
					elist = AST.ExprListNode(elist + [e.left, create_div(1,e.right)])
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
					temp = self.visitFuncCall(fcall, True)
					exp = temp
					return exp
				else:
					return e


	def visitMap(self, node: AST.MapNode):
		val = self.visit(node.expr)
		output = self.get_map(val, node)
		return output 

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
			if(str(e) in self.arrayLens):
				self.arrayLens[str(out)] = self.arrayLens[str(e)]
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
		expr = self.visit(node.expr)
		constraints = self.visit(node.constraints)

		if(str(constraints) in self.arrayLens):
			x = self.arrayLens[str(constraints)]
			for i in range(len(constraints)):
				constraints[i] = OR(constraints[i],LT(x, i+1))

		out = Real('Lp_'+str(self.number.nextn()))
		if(node.op == "maximize"):
			self.C.append(Implies(self.convertToZ3(constraints), out >= self.convertToZ3(expr)))
		else:
			self.C.append(Implies(self.convertToZ3(constraints), out <= self.convertToZ3(expr)))

		return out


	def visitFunc(self, node: AST.FuncNode):
		name = node.decl.name.name
		self.F[name] = node

	def visitSeq(self, node: AST.SeqNode):
		self.visit(node.stmt1)
		self.visit(node.stmt2)
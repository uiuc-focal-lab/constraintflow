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

	def __init__(self, store, F, M, V, C, shape):
		self.F = F #{function name -> function info}
		self.M = M #{(op, value) -> value}
		self.V = V #{symbolic var -> vertex}
		self.C = C #symbolic constraints
		self.store = store #{var -> value}
		self.shape = shape #{name -> type} of each variable in the shape
		self.vname = 0
		self.limit = 3 #max 3 neurons are connected to each one
		self.number = Number()
		self.symb = True

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

	def ADDPoly(self, left, right):
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

		c = (left.const[0] + right.const[0], "Float")
		n = {}
		for leftn in left.coeffs.keys():
			if(leftn in right.coeffs.keys()):
				n[leftn] = (left.coeffs[leftn][0] + right.coeffs[leftn][0], "Float")
			else:
				n[leftn] = (left.coeffs[leftn][0], "Float")

		for rightn in right.coeffs.keys():
			if(not rightn in n.keys()):
				n[rightn] = (right.coeffs[rightn][0], "Float")

		return PolyExpValue(n, c)

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

		c = left.const - right.const
		n = {}
		for leftn in left.coeffs.keys():
			if(leftn in right.coeffs.keys()):
				n[leftn] = (left.coeffs[leftn][0] - right.coeffs[leftn][0], "Float")
			else:
				n[leftn] = (left.coeffs[leftn][0], "Float")

		for rightn in right.coeffs.keys():
			if(not rightn in n.keys()):
				n[rightn] = (- right.coeffs[rightn][0], "Float")

		return PolyExpValue(n, c)

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
			if(node.left[1] == "Neuron"):
				return PolyExpValue({node.left: node.right}, (0, "Float"))
			else:
				return PolyExpValue({node.right: node.left}, (0, "Float"))
		elif(isinstance(node, DIV)):
			return PolyExpValue({node.left: (1 / node.right[0], "Float")}, (0, "Float"))
		else:
			print(node)
			assert False

	def convertToZ3(self, node):

		if(isinstance(node, tuple)):
			return node[0]
		elif(isinstance(node, ADD)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			return l + r
		elif(isinstance(node, SUB)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			return l - r
		elif(isinstance(node, MULT)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			return l * r
		elif(isinstance(node, DIV)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			return l / r
		elif(isinstance(node, NEG)):
			l = self.convertToZ3(node.left)
			return -l
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
			return And(l, r)
		elif(isinstance(node, OR)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			return Or(l,r)
		elif(isinstance(node, IF)):
			c = self.convertToZ3(node.cond)
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			return If(c, l, r)
		else:
			return node

	def get_binop(self, left, right, f):
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
		return self.M[Epsilon(node.identifier)]
	
	def visitExprList(self, node: AST.ExprListNode):
		exps = []
		for exp in node.exprlist:
			exps.append(self.visit(exp))
		return exps

	def visitBinOp(self, node: AST.BinOpNode):
		left = self.visit(node.left)
		right = self.visit(node.right)
		else:
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
			else:	
				pre_elist = self.visit(node.func)
			n = len(elist)
			temp_array = []
			for i in range(n):
				temp_array += [None]*n 
			for i in range(n):
				for j in range(n):
					if i == j:
						temp_array[i][j] = (True, 'Bool')
					else:
						if(node.op == "argmax"):
							arglist = AST.ExprListNode(pre_elist + [elist[i], elist[j]])
						else:
							arglist = AST.ExprListNode(pre_elist + [elist[j], elist[i]])
						fcall = AST.FuncCallNode(node.func, arglist)
						temp_array[i][j] = self.visit(fcall)
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
				temp_array += [None]*n 
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
		return IF(self.get_binop(e1, e2, GEQ), e1, e2)

	def get_listOp(self, elist, node):
		if isinstance(elist, IF):
			return IF(elist.cond, self.get_listOp(elist.left, node), self.get_listOp(elist.right, node))
		else:
			sum = (0, "Float")
			length = (len(elist), 'Int')
			for e in elist:
				sum = self.get_binop(sum, e, ADD)
			if node.op == 'sum':
				return sum
			elif node.op == 'len':
				return length 
			elif node.op == 'avg':
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

	def visitDot(self, node: AST.DotNode):
		left = self.visit(node.left)
		right = self.visit(node.right)
		return self.get_dot(left, right)

	def get_getElement(self, expr, name):
		if isinstance(expr, IF):
			return IF(expr.cond, self.get_getElement(expr.left, name), self.get_getElement(expr.right, name))
		if isinstance(expr, list):
			out = []
			for e in expr:
				out += self.get_getElement(e, name)
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

	def get_map(self, e, node):
		if isinstance(e, IF):
			return IF(e.cond, self.get_map(e.left, node), self.get_map(e.right, node))
		else:
			epoly = self.convertToPoly(e)
			exp = epoly.const
			for n in epoly.coeffs.keys():

				if isinstance(node.func, AST.VarNode):
					elist = []
				else:	
					elist = self.visit(node.func)
				elist = AST.ExprListNode(elist + [n, epoly.coeffs[n]])
				fcall = AST.FuncCallNode(node.func.name, elist)
				exp = self.get_binop(exp, self.visitFuncCall(fcall, True), ADD)

			return exp

	def visitMap(self, node: AST.MapNode):
		e = self.visit(node.expr)
		return self.get_map(e, node)

	def get_get_mapList(self, e, node):
		if isinstance(e, IF):
			return IF(e.cond, self.get_get_mapList(e.left, node), self.get_get_mapList(self.right, node))
		else:
			if isinstance(node.func, AST.VarNode):
				elist = []
			else:	
				elist = self.visit(node.func)
			elist = AST.ExprListNode(elist + [e])
			fcall = AST.FuncCallNode(node.func.name, elist)
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

		return self.M[Traverse(e, node.direction, p_name, s_name, node.func.name)]

	# def visitCurr(self, node: AST.CurrNode):
	# 	return self.store["curr"]
	
	# def visitPrev(self, node: AST.PrevNode):
	# 	return self.store["prev"]

	'''
	



	def visitShapeDecl(self, node: AST.ShapeDeclNode):

	def visitTransRetBasic(self, node: AST.TransRetBasicNode):



	def visitTransRetIf(self, node: AST.TransRetIfNode):


	def visitOpStmt(self, node: AST.OpStmtNode):

		
	def visitOpList(self, node: AST.OpListNode):


	def visitTransformer(self, node: AST.TransformerNode):

	'''

	

				

	def visitPropTermOp(self, prop):
		left = self.convertToZ3(self.visit(prop.leftpt))
		right = self.convertToZ3(self.visit(prop.rightpt))
		if(prop.op == "+"):
			return left + right
		elif(prop.op == "-"):
			return left - right

	def visitDoubleProp(self, prop):
		left = self.visit(prop.leftprop)
		right = self.visit(prop.rightprop)
		if(prop.op == "and"):
			return And(left, right)
		if(prop.op == "or"):
			return Or(left, right)

	def visitPropTermBasic(self, prop):
		return self.convertToZ3(self.visit(prop.term))

	def visitSingleProp(self, pt):
		left = self.convertToZ3(self.visit(pt.leftpt))
		right = self.convertToZ3(self.visit(pt.rightpt))
		if(pt.op == "<"):
			return left < right
		elif(pt.op == "<="):
			return left <= right
		elif(pt.op == ">"):
			return left > right
		elif(pt.op == ">="):
			return left >= right
		elif(pt.op == "=="):
			return left == right

	def visitPropTermIn(self, pt):
		nexpr = self.visit(pt.n)
		zexpr = self.visit(pt.z)
		#Calculate min and max of zexpr
	

	def visitFunc(self, node: AST.FuncNode):
		name = node.decl.name.name
		self.F[name] = node

	def visitSeq(self, node: AST.SeqNode):
		self.visit(node.stmt1)
		self.visit(node.stmt2)

	def visitFlow(self, node: AST.FlowNode):
		pass #verification should be done by this point

	def visitProg(self, node: AST.ProgramNode):
		self.visit(node.shape)
		self.visit(node.stmt)

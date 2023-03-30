import astVisitor
import ast as AST
from z3 import *
from value import *
import astPrinter

class Number:

	def __init__(self):
		self.i = 0

	def nextn(self):
		self.i = self.i + 1
		return self.i

class Vertex:
	#Store the expression for the vertex polyhedral element when it is defined(say if we are using a map function)
	def __init__(self, name):
		self.symmap = {} #ex) "layer" -> Int('X1')
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
		self.coeffs = coeffs #List of (coeff, op)

	# def __eq__(self, obj):
	# 	if(isinstance(obj, ZonoExpValue)):
	# 		return self.coeffs == obj.coeffs 
	# 	else:
	# 		return False


#Most of the visit functions return a value. 
#The property visit functions return a symbolic constraint
class SymbolicOperationalSemantics(astVisitor.ASTVisitor):

	def __init__(self, store, F, M, V, C, shape):
		self.FMap = F #{function name -> function info}
		self.V = V #{symbolic var -> vertex}
		self.M = M #{(op, value) -> value}
		self.store = store #{var -> value}
		self.C = C #symbolic constraints
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

	def visitExprList(self, node: AST.ExprListNode):
		exps = []
		for exp in node.exprlist:
			exps.append(self.visit(exp))
		return exps
		
	
	def visitInt(self, node: AST.ConstIntNode):
		return (node.value, "Int")

	def visitFloat(self, node: AST.ConstFloatNode):
		return (node.value, "Float")
	
	def visitBool(self, node: AST.ConstBoolNode):
		return (node.value, "Bool")

	def ConstToPoly(self, const):
		return PolyExpValue({}, const)

	def NeuronToPoly(self, n):
		return PolyExpValue({n: (1, "Float")}, (0, "Float"))

	def AddPoly(self, left, right):
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

	def SubPoly(self, left, right):
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

		c = left.const + right.const
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

	def visitBinOp(self, node: AST.BinOpNode):
		left = self.visit(node.left)
		right = self.visit(node.right)
		if(node.op == "+"):
			return Add(left, right)
		elif(node.op == "-"):
			return Sub(left, right)
		elif(node.op == "*"):
			return Mult(left, right)
		elif(node.op == "/"):
			return Div(left, right)
		elif(node.op == "<="):
			return LEQ(left, right)
		elif(node.op == "<"):
			return LT(left, right)
		elif(node.op == ">="):
			return GEQ(left, right)
		elif(node.op == ">"):
			return GT(left, right)
		elif(node.op == "and"):
			return AND(left, right)
		elif(node.op == "or"):
			return OR(left, right)
		elif(node.op == "=="):
			return EQQ(left, right)

	def visitUnOp(self, node: AST.UnOpNode):
		expr = self.visit(node.expr)
		if(node.op == "-"):
			return NEG(expr)
		elif(node.op == "~"):
			return NOT(expr)

	def visitTernary(self, node: AST.TernaryNode):
		cond = self.visit(node.cond)
		left = self.visit(node.texpr)
		right = self.visit(node.fexpr)
		return self.M[Ternary(cond, left, right)]

	def visitNlistOp(self, node: AST.NlistOpNode):
		if(node.op == "min"):
			e = self.visit(node.expr)
			return self.M[MIN(e)]
		elif(node.op == "max"):
			e = self.visit(node.expr)
			return self.M[MAX(e)]
		elif(node.op == "argmin"):
			e = self.visit(node.expr)
			return self.M[ARGMIN(e, node.elem.name)]
		elif(node.op == "argmax"):
			e = self.visit(node.expr)
			return self.M[ARGMAX(e, node.elem.name)]

	def visitSum(self, node: AST.SumNode):
		elist = self.visit(node.expr)
		sum = (0, "Float")
		for e in elist:
			sum = Add(sum, e)
		return sum

	def visitDot(self, node: AST.DotNode):
		left = self.visit(node.left)
		right = self.visit(node.right)

		sum = (0, "Float")
		for i in range(min(len(left), len(right))):
			sum = Add(sum, Mult(left[i],right[i]))

		return sum

	def visitFuncCall(self, node: AST.FuncCallNode, preeval = False):
		func = self.FMap[node.name.name]

		newvars = []
		oldvalues = {}

		if(not preeval):
			elist = self.visit(node.arglist)
		else:
			elist = node.arglist.exprlist

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

	def convertToPoly(self, node):
		
		if(isinstance(node, tuple)):
			if(node[1] == "Float" or node[1] == "Int"):
				return PolyExpValue({}, node)
			elif(node[1] == "Neuron"):
				return PolyExpValue({node: (1, "Float")}, (0, "Float"))
		elif(isinstance(node, Add)):
			left = self.convertToPoly(node.left)
			right = self.convertToPoly(node.right)
			return self.AddPoly(left, right)
		elif(isinstance(node, Sub)):
			left = self.convertToPoly(node.left)
			right = self.convertToPoly(node.right)
			return self.SubPoly(left, right)
		elif(isinstance(node, Mult)):
			if(node.left[1] == "Neuron"):
				return PolyExpValue({node.left: node.right}, (0, "Float"))
			else:
				return PolyExpValue({node.right: node.left}, (0, "Float"))
		elif(isinstance(node, Div)):
			return PolyExpValue({node.left: (1 / node.right[0], "Float")}, (0, "Float"))
		else:
			print(node)
			assert False

	def convertToZ3(self, node):

		if(isinstance(node, tuple)):
			return node[0]
		elif(isinstance(node, Add)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			return l + r
		elif(isinstance(node, Sub)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			return l - r
		elif(isinstance(node, Mult)):
			l = self.convertToZ3(node.left)
			r = self.convertToZ3(node.right)
			return l * r
		elif(isinstance(node, Div)):
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
		else:
			return node

	def visitMap(self, node: AST.MapNode):
		e = self.visit(node.expr)
		epoly = self.convertToPoly(e)

		exp = epoly.const
		for n in epoly.coeffs.keys():
			elist = AST.ExprListNode([n, epoly.coeffs[n]])
			fcall = AST.FuncCallNode(node.func, elist)
			exp = Add(exp, self.visitFuncCall(fcall, True))

		return exp

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

	def visitSub(self, node: AST.SubNode):
		le = self.visit(node.listexpr)
		e = self.visit(node.expr)
		return self.M[LISTSUB(le, e)]

	

	def visitCurr(self, node: AST.CurrNode):
		return self.store["curr"]
	
	def visitPrev(self, node: AST.PrevNode):
		return self.store["prev"]

	'''
	def visitEpsilon(self, node: AST.EpsilonNode):



	def visitShapeDecl(self, node: AST.ShapeDeclNode):

	def visitTransRetBasic(self, node: AST.TransRetBasicNode):



	def visitTransRetIf(self, node: AST.TransRetIfNode):


	def visitOpStmt(self, node: AST.OpStmtNode):

		
	def visitOpList(self, node: AST.OpListNode):


	def visitTransformer(self, node: AST.TransformerNode):

	'''

	def visitGetElement(self, pt):
		n = self.visit(pt.expr)
		if(isinstance(n, list)):
			return [self.V[i[0]].symmap[pt.elem.name] for i in n]
		else:
			return self.V[n[0]].symmap[pt.elem.name]

	def visitGetMetadata(self, pt):
		n = self.visit(pt.expr)
		if(isinstance(n, list)):
			return [self.V[i[0]].symmap[pt.metadata.name] for i in n]
		else:
			return self.V[n[0]].symmap[pt.metadata.name]

	def visitVar(self, pt):
		return self.store[pt.name]			

	def visitPropTermOp(self, prop):
		left = self.visit(prop.leftpt)
		right = self.visit(prop.rightpt)
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
		if(isinstance(prop.term, AST.CurrNode)):
			return self.store["curr'"]
		else:
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
		self.FMap[name] = node

	def visitSeq(self, node: AST.SeqNode):
		self.visit(node.stmt1)
		self.visit(node.stmt2)

	def visitFlow(self, node: AST.FlowNode):
		pass #verification should be done by this point

	def visitProg(self, node: AST.ProgramNode):
		self.visit(node.shape)
		self.visit(node.stmt)

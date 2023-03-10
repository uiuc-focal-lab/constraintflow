import astVisitor
import ast as AST
from z3 import *
from value import *

class Number:

	def __init__(self):
		self.i = 0

	def nextn(self):
		self.i = self.i + 1
		return self.i

#We can have V be a set of vertices. Each vertex is the mapping of itself and all of it's metadata/shape data to symbolic variables.
#M will be the mapping of values to metadata. So we can have argmax(prev, u) + max(curr[l]) -> x34, curr[l] -> x23, argmax(prev, u) -> x25. 
#The constraints between all of the things mapped in M will be added directly to C when things are being added to M.
#Same with V, because when we add 3 vertices for prev, we will add the constraints that the property holds for all of them(from the apply function)

class Vertex:
	#Store the expression for the vertex polyhedral element when it is defined(say if we are using a map function)
	def __init__(self, name, shape, number):
		self.symmap = {}
		#self.symmap[name] = Real('X' + str(number.nextn()))
		self.name = Real('X' + str(number.nextn()))
		'''
		self.symmap["layer"] = Int('X' + str(number.nextn()))
		self.symmap["weight"] = []
		self.symmap["bias"] = Real('X' + str(number.nextn()))
		for (t, s) in shape:
			if(t == "Int"):
				map[s] = Int('X' + str(number.nextn()))
			else:
				map[s] = Real('X' + str(number.nextn()))
		self.expmap = {} #will be L -> c0 + c1 * v1 if needed
		'''

class Add:

	def __init__(self, left, right):
		self.left = left
		self.right = right

class Sub:

	def __init__(self, left, right):
		self.left = left
		self.right = right

class Mult:

	def __init__(self, left, right):
		self.left = left
		self.right = right

class Div:

	def __init__(self, left, right):
		self.left = left
		self.right = right

class LEQ:

	def __init__(self, left, right):
		self.left = left
		self.right = right

class LT:

	def __init__(self, left, right):
		self.left = left
		self.right = right

class GEQ:

	def __init__(self, left, right):
		self.left = left
		self.right = right

class GT:

	def __init__(self, left, right):
		self.left = left
		self.right = right

class EQQ:

	def __init__(self, left, right):
		self.left = left
		self.right = right

class NEQ:

	def __init__(self, left, right):
		self.left = left
		self.right = right

class AND:

	def __init__(self, left, right):
		self.left = left
		self.right = right

class OR:

	def __init__(self, left, right):
		self.left = left
		self.right = right

class NOT:

	def __init__(self, right):
		self.right = right

class NEG:

	def __init__(self, right):
		self.right = right

class Ternary:

	def __init__(self, cond, left, right):
		self.cond = cond
		self.left = left
		self.right = right

class MAX:

	def __init__(self, e):
		self.e = e

class MIN:

	def __init__(self, e):
		self.e = e

class ARGMAX:

	def __init__(self, e, s):
		self.e = e
		self.s = s

class ARGMIN:

	def __init__(self, e, s):
		self.e = e
		self.s = s

class LISTSUB:

	def __init__(self, l, e):
		self.l = l
		self.s = s

class Traverse:

	def __init__(self, e, d, f1, f2, f3):
		self.e = e
		self.d = d
		self.f1 = f1
		self.f2 = f2
		self.f3 = f3


class ApplyShapeProp(astVisitor.ASTVisitor):

	def visitCurr(self, pt, v, M):
		return M[v]

	def visitGetElement(self, pt, v, M):
		return M[pt]

	def visitGetMetadata(self, pt, v, M):
		return M[pt]

	def visitSingleProp(self, prop, v, M):
		left = self.visit(prop.leftpt)
		right = self.visit(prop.rightpt)
		if(prop.op == "+"):
			return left + right
		elif(prop.op == "-"):
			return left - right

	def visitDoubleProp(self, prop, v, M):
		left = self.visit(prop.leftprop, v, M)
		right = self.visit(prop.rightprop, v, M)
		if(prop.op == "and"):
			return And(left, right)
		if(prop.op == "left"):
			return Or(left, right)

	def visitPropTermBasic(self, pt, v, M):
		return self.visit(prop.term)

	def visitPropTermOp(self, pt, v, M):
		left = self.visit(pt.leftpt)
		right = self.visit(pt.rightpt)
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

	def visitPropTermIn(self, pt, v, M):
		nexpr = self.visit(pt.n)
		zexpr = self.visit(pt.z)
		#Calculate min and max of zexpr


def apply(prop: AST.PropNode, v, M):
	ApplyShapeProp().visit(prop, v, M)

def apply(prop: AST.PropNode, store, M, V):
	ApplyTraverseProp().visit(prop, store, M, V)


#Most of the visit functions return a value. 
#The property visit functions return a symbolic constraint?
class Evaluate(astVisitor.ASTVisitor):

	def __init__(self, store, F, M, V, C, shape):
		self.FMap = F #{function name -> function info}
		self.V = V #{symbolic var -> vertex}
		self.M = M #{(op, value) -> value}
		self.store = store #{var -> value}
		self.C = C #symbolic constraints
		self.shape = self.shape #(type, name) of each variable in the shape
		self.constraint = None
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
	

	def visitExprList(self, node: AST.ExprListNode):
		exps = []
		for exp in node.exprlist:
			exps.append(self.visit(exp))
		return exps		
		

		
	def visitUnOp(self, node: AST.UnOpNode):
		

		
	def visitNeuron(self, node: AST.NeuronNode):
		return NeuronValue(node.name)
	'''
	def visitInt(self, node: AST.ConstIntNode):
		return (node.value, "Int")

	def visitFloat(self, node: AST.ConstFloatNode):
		return (node.value, "Float")
	
	def visitBool(self, node: AST.ConstBoolNode):
		return (node.value, "Bool")

	def ConstToPoly(self, const):
		return PolyExpValue({}, const)

	def NeuronToPoly(self, n):
		return PolyExpValue({n: (1, "+")}, 0)

	def AddPoly(self, left, right):
		c = left.const + right.const
		n = {}
		for leftn in left.coeffs.keys():
			if(leftn in right.coeffs.keys()):
				n[leftn] = left.coeffs[leftn][0] + right.coeffs[leftn][0]
			else:
				n[leftn] = left.coeffs[leftn][0]

		for rightn in right.coeffs.keys():
			if(not rightn in n.keys()):
				n[rightn] = right.coeffs[rightn][0]

		p = PolyExpValue(n, c)

	def SubPoly(self, left, right):
		c = left.const + right.const
		n = {}
		for leftn in left.coeffs.keys():
			if(leftn in right.coeffs.keys()):
				n[leftn] = left.coeffs[leftn][0] - right.coeffs[leftn][0]
			else:
				n[leftn] = left.coeffs[leftn][0]

		for rightn in right.coeffs.keys():
			if(not rightn in n.keys()):
				n[rightn] = - right.coeffs[rightn][0]

		p = PolyExpValue(n, c)

	def visitBinOp(self, node: AST.BinOpNode):
		left = self.visit(node.left)
		right = self.visit(node.left)

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

		'''
			if(node.op == "*"):
				if(isinstance(left, Vertex)):
					return PolyExpValue({left: (right, "+")},0)
				else:
					return PolyExpValue({right: (left, "+")},0)
			elif(node.op == "/"):
				return PolyExpValue({left: (1 / right, "+")},0)
			elif(node.op == "+"):
				if(isinstance(left, Vertex) or isinstance(right, Vertex) or isinstance(left, PolyExpValue) or isinstance(right, PolyExpValue)):
					if(isinstance(left, Vertex)):
						left = self.NeuronToPoly(left)
					else:
						left = self.ConstToPoly(left)

					if(isinstance(right, Vertex)):
						right = self.NeuronToPoly(right)
					else:
						right = self.ConstToPoly(right)

					self.AddPoly(left, right)
				else:
					return left + right
			elif(node.op == "-"):
				if(isinstance(left, Vertex) or isinstance(right, Vertex) or isinstance(left, PolyExpValue) or isinstance(right, PolyExpValue)):
					if(isinstance(left, Vertex)):
						left = self.NeuronToPoly(left)
					else:
						left = self.ConstToPoly(left)

					if(isinstance(right, Vertex)):
						right = self.NeuronToPoly(right)
					else:
						right = self.ConstToPoly(right)

					self.SubPoly(left, right)
				else:
					return left - right
			'''

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
		return self.M[IF(cond, left, right)]

	def visitNlistOp(self, node: AST.NlistOpNode):
		if(node.op == "min"):
			e = self.visit(node.expr)
			return self.M[MIN(e)]
		elif(node.op == "max"):
			e = self.visit(node.expr)
			return self.M[MAX(e)]
		elif(node.op == "argmin"):
			e = self.visit(node.expr)
			element = self.visit(node.elem)
			return self.M[ARGMIN(e, element)]
		elif(node.op == "argmax"):
			e = self.visit(node.expr)
			element = self.visit(node.elem)
			return self.M[ARGMAX(e, element)]

	def visitSum(self, node: AST.SumNode):
		elist = self.visit(node.expr)
		sum = 0
		for e in elist:
			sum = Add(sum, e)
		return sum

	def visitDot(self, node: AST.DotNode):
		left = self.visit(node.left)
		right = self.visit(node.right)
		sum = 0
		for i in range(min(len(left), len(right))):
			sum = Add(sum, Mult(left[i],right[i]))

		return sum

	def visitFuncCall(self, node: AST.FuncCallNode):
		func = self.FMap(node.name)

		newvars = []
		oldvalues = {}
		for (exp,(t, arg)) in zip(self.visit(node.arglist), func.decl.arglist.arglist):
			if arg in self.store.keys():
				oldvalues[arg] = self.store[arg]
			else:
				newvars.append(arg)

			store[arg] = exp

		val = self.visit(node.expr)
		
		for v in newvars:
			del self.store[v]

		for ov in oldvalues.keys():
			self.store[ov] = oldvalues[ov]

		return val

	def convertToPoly(self, node):
		
		if(isinstance(node, tuple)):
			if(node[1] == "Float" or node[1] == "Int"):
				PolyExpValue({}, node[0])
			elif(node[1] == "Neuron"):
				PolyExpValue({node[0]: 1}, 0)
		elif(isinstance(node, Add)):
			left = convertToPoly(node.left)
			right = convertToPoly(node.right)
			return AddPoly(left, right)
		elif(isinstance(node, Sub)):
			left = convertToPoly(node.left)
			right = convertToPoly(node.right)
			return SubPoly(left, right)
		elif(isinstance(node, Mult)):
			if(node.left[1] == "Neuron"):
				PolyExpValue({node.left[0]: node.right[0]}, 0)
			else:
				PolyExpValue({node.right[0]: node.left[0]}, 0)
		elif(isinstance(node, Div)):
			PolyExpValue({node.left[0]: 1 / node.right[0]}, 0)

	def convertToZ3(self, node):

		if(isinstance(node, tuple)):
			return node[0]
		elif(isinstance(node, Add)):
			l = convertToZ3(node.left)
			r = convertToZ3(node.right)
			return l + r
		elif(isinstance(node, Sub)):
			l = convertToZ3(node.left)
			r = convertToZ3(node.right)
			return l - r
		elif(isinstance(node, Mult)):
			l = convertToZ3(node.left)
			r = convertToZ3(node.right)
			return l * r
		elif(isinstance(node, Div)):
			l = convertToZ3(node.left)
			r = convertToZ3(node.right)
			return l / r
		elif(isinstance(node, NEG)):
			l = convertToZ3(node.left)
			return -l

	def visitMap(self, node: AST.MapNode):
		e = self.visit(node.expr)
		epoly = self.convertToPoly(e)

		exp = epoly.const
		for n in epoly.coeffs.keys():
			elist = ExprList([n, epoly.coeffs[n]])
			fcall = AST.FuncCallNode(node.func, elist)
			exp = Add(exp, self.visit(fcall))

		return exp

	def visitTraverse(self, node: AST.TraverseNode):
		e = self.visit(node.expr)
		return self.M[Traverse(e, node.direction, node.priority, node.stop, node.func)]

	def visitSub(self, node: AST.SubNode):
		le = self.visit(node.listexpr)
		e = self.visit(node.expr)
		return self.M[LISTSUB(le, e)]

	'''

	def visitCurr(self, node: AST.CurrNode):
		if(not )
	
	def visitPrev(self, node: AST.PrevNode):
		if(not "prev" in store.keys()):
			nlist = []
			for i in range(self.limit):
				name = self.getVname()
				V[name] = Vertex(name, shape, number)
				nlist.add(Neuron(name))
			store['prev'] = nlist

		return store['prev']


	def visitEpsilon(self, node: AST.EpsilonNode):



	def visitShapeDecl(self, node: AST.ShapeDeclNode):
		for (t, v) in node.elements.arglist:
			self.varTypes[v.name()] = t
			self.shape.append(v.name())
		self.constraint = node.p

	def visitTransRetBasic(self, node: AST.TransRetBasicNode):



	def visitTransRetIf(self, node: AST.TransRetIfNode):


	def visitOpStmt(self, node: AST.OpStmtNode):

		
	def visitOpList(self, node: AST.OpListNode):


	def visitTransformer(self, node: AST.TransformerNode):

	'''

	def visitGetElement(self, pt):
		n = self.visit(pt.expr)
		if(isinstance(n, list)):
			return [self.V[i[0]].symmap[pt.elem] for i in n]
		else:
			return self.V[n[0]].symmap[pt.elem]

	def visitGetMetadata(self, pt):
		n = self.visit(pt.expr)
		if(isinstance(n, list)):
			return [self.V[i[0]].symmap[pt.metadata] for i in n]
		else:
			return self.V[n[0]].symmap[pt.metadata]

	def visitVar(self, pt):
		return self.store[pt.name]
			
		
		'''
		elif(isinstance(e, PolyExpValue)):
				exp = e.const
				for n in e.coeffs.keys():

					(c, op) = e.coeffs[n]
					if(op == "+"):
						exp = exp + (n.name * c)
						
					else:
						exp = exp - (n.name * c)
				return exp
		'''

		#For variables representing function arguments. 
		#For variables representing functions, we will do something different.
		#We will not call this function in the latter case bc we'll know it is a function call.

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
		return self.visit(prop.term)

	def visitSingleProp(self, pt):
		left = self.visit(pt.leftpt)
		right = self.visit(pt.rightpt)
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

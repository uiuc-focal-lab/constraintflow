import ASTVisitor
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
		self.symmap[name] = Real('X' + str(number.nextn()))
		self.symmap["layer"] = Int('X' + str(number.nextn()))
		self.symmap["weight"] = []
		self.symmap["bias"] = Real('X' + str(number.nextn()))
		for (t, s) in shape:
			if(t == "Int"):
				map[s] = Int('X' + str(number.nextn()))
			else:
				map[s] = Real('X' + str(number.nextn()))
		self.expmap = {} #will be L -> c0 + c1 * v1 if needed

class VSet:

	def __init__(self):
		#vertex name to vertex
		self.vertices = {}


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

class ApplyTraverseProp(astVisitor.ASTVisitor):

	def visitGetElement(self, pt, store, M, V):
		#evaulate pt.expr and get a Neuron name or Neuron List name -> n
		#Evaluating on prev will generate a list of new neurons and return them
		#Evaluating on argmax(prev, l) will create a new neuron, add constraints to C and return the new neuron's name
		if(isinstance(n, list)):
			xlist = [V.vertices[xn].symmap[pt.elem] for xn in n]
		else:
			return V.vertices[n].symmap[pt.elem]

	def visitGetMetadata(self, pt, store, M, V):
		#evaulate pt.expr and get a Neuron name or Neuron List name -> n
		#Evaluating on prev will generate a list of new neurons and return them
		#Evaluating on argmax(prev, l) will create a new neuron, add constraints to C and return the new neuron's name
		if(isinstance(n, list)):
			xlist = [V.vertices[xn].symmap[pt.metadata] for xn in n]
		else:
			return V.vertices[n].symmap[pt.metadata]

	def visitVar(self, pt, store, M, V):
		e = store[pt.name()]
		if(isinstance(e, AST.Neuron)):
			return V[e]
		else:
			return M[e]

	def visitSingleProp(self, prop, store, M, V):
		left = self.visit(prop.leftpt)
		right = self.visit(prop.rightpt)
		if(prop.op == "+"):
			return left + right
		elif(prop.op == "-"):
			return left - right

	def visitDoubleProp(self, prop, store, M, V):
		left = self.visit(prop.leftprop, v, M)
		right = self.visit(prop.rightprop, v, M)
		if(prop.op == "and"):
			return And(left, right)
		if(prop.op == "left"):
			return Or(left, right)

	def visitPropTermBasic(self, prop, store, M, V):
		return self.visit(prop.term)

	def visitPropTermOp(self, prop, store, M, V):
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

	def visitPropTermIn(self, pt, v, M, V):
		nexpr = self.visit(pt.n)
		zexpr = self.visit(pt.z)
		#Calculate min and max of zexpr

def apply(prop: AST.PropNode, v, M):
	ApplyShapeProp().visit(prop, v, M)

def apply(prop: AST.PropNode, store, M, V):
	ApplyTraverseProp().visit(prop, store, M, V)

class Evaluate(astVisitor.ASTVisitor):

	def __init__(self):
		self.varTypes = {}
		self.varMap = {}
		self.V = {}
		self.M = {}
		self.C = []
		self.shape = []
		self.constraint = None

	def visitBaseType(self, node: AST.BaseTypeNode):
	

	def visitArrayType(self, node: AST.ArrayTypeNode):
	

	def visitArgList(self, node: AST.ArgListNode):
	

	def visitExprList(self, node: AST.ExprListNode):
		exps = []
		for exp in node.exprlist:
			exps.append(self.visit(exp))
		return exps		

	def visitBinOp(self, node: AST.BinOpNode):
		

		
	def visitUnOp(self, node: AST.UnOpNode):
		

	def visitNlistOp(self, node: AST.NlistOpNode):
		

	def visitVar(self, node: AST.VarNode):
		
	def visitNeuron(self, node: AST.NeuronNode):
	
	def visitInt(self, node: AST.ConstIntNode):
		return IntValue(node.value)

	def visitFloat(self, node: AST.ConstFloatNode):
		return FloatValue(node.value)
	
	def visitBool(self, node: AST.ConstBoolNode):
		return BoolValue(node.value)

	def visitCurr(self, node: AST.CurrNode):
	
	def visitPrev(self, node: AST.PrevNode):


	def visitEpsilon(self, node: AST.EpsilonNode):
	
	#For general if statements, we can map the true branch expr to some symbolic variable, v1
	#Then, map the false banch expr to some symbolic variable, v2
	#Then, map the conditional expr to some symbolic variable, v3
	#Then, map the whole expression to some symbolic variable, v4
	#Then, in the constraints, we will say (v3 and (v4 == v1)) or (not v3 and (v4 == v2))
	def visitTernary(self, node: AST.TernaryNode):


	def visitGetMetadata(self, node: AST.GetMetadataNode):


	def visitGetElement(self, node: AST.GetElementNode):



	def visitTraverse(self, node: AST.TraverseNode):


	def visitSum(self, node: AST.SumNode):


	def visitSub(self, node: AST.SubNode):


	def visitMap(self, node: AST.MapNode):



	def visitDot(self, node: AST.DotNode):


	def visitFuncCall(self, node: AST.FuncCallNode):


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


	def visitPropTermBasic(self, node: AST.PropTermBasicNode):

	def visitPropTermIn(self, node: AST.PropTermInNode):


	def visitPropTermOp(self, node: AST.PropTermOpNode):


	def visitSingleProp(self, node: AST.SinglePropNode):
	
	def visitDoubleProp(self, node: AST.DoublePropNode):
	

	def visitFunc(self, node: AST.FuncNode):
		name = node.decl.name.name
		self.varMap[name] = node

	def visitSeq(self, node: AST.SeqNode):
		self.visit(node.stmt1)
		self.visit(node.stmt2)

	def visitFlow(self, node: AST.FlowNode):
		pass #verification should be done by this point

	def visitProg(self, node: AST.ProgramNode):
		self.visit(node.shape)
		self.visit(node.stmt)
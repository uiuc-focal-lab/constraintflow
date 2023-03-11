import astVisitor
import ast as AST
from z3 import *
from value import *
from verify import *
import copy


class SymbolicGraph(astVisitor.ASTVisitor):

	def __init__(self, store, F, constraint, shape):
		self.M = {}
		self.V = {}
		self.C = []
		self.constraint = constraint
		self.store = store
		self.F = F
		self.shape = shape
		self.os = Evaluate(self.store, self.F, self.M, self.V, self.C, self.shape)
		self.number = Number()
		self.N = 3

	def visitInt(self, node):
		pass

	def visitVar(self, node):
		pass

	def visitFloat(self, node):
		pass

	def visitBool(self, node):
		pass

	def visitBinOp(self, node):
		self.visit(node.left)
		self.visit(node.right)

	def visitUnOp(self, node):
		self.visit(node.expr)

	def visitSum(self, node):
		self.visit(node.expr)

	def visitDot(self, node):
		self.visit(left)
		self.visit(right)

	def visitPrev(self, node):
		pass

	def visitCurr(self, node):
		pass

	def visitGetMetadata(self, node):
		self.visit(node.expr)
		n = self.os.visit(node.expr)
		if(not isinstance(n, list)):
			if not node.metadata in self.V[n[0]].symmap.keys():
				newvar = None
				if(node.metadata == "bias"):
					newvar = (Real('X' + str(self.number.nextn())), "Bool")
				elif(node.metadate == "weight"):
					newvar = [(Real('X' + str(self.number.nextn())), "Float") for i in self.N]
				elif(node.metadata == "layer"):
					newvar = (Int('X' + str(self.number.nextn())), "Int")
				self.V[n[0]].symmap[node.metadata] = newvar
		else:
			for ni in n:
				if not node.metadata in self.V[ni[0]].symmap.keys():
					newvar = None
					if(node.metadata == "bias"):
						newvar = (Real('X' + str(self.number.nextn())), "Bool")
					elif(node.metadate == "weight"):
						newvar = [(Real('X' + str(self.number.nextn())), "Float") for i in self.N]
					elif(node.metadata == "layer"):
						newvar = (Int('X' + str(self.number.nextn())), "Int")
					self.V[ni[0]].symmap[node.metadata] = newvar

	def visitGetElement(self, node):
		self.visit(node.expr)
		n = self.os.visit(node.expr)
		if(not isinstance(n, list)):
			if not node.elem.name in self.V[n[0]].symmap.keys():
				newvar = None
				if(self.shape[node.elem.name] == "Bool"):
					newvar = (Bool('X' + str(self.number.nextn())), "Bool")
				elif(self.shape[node.elem.name] == "Int"):
					newvar = (Int('X' + str(self.number.nextn())), "Int")
				else:
					newvar = (Real('X' + str(self.number.nextn())), "Float")
				self.V[n[0]].symmap[node.elem.name] = newvar
		else:
			for ni in n:
				if not node.elem.name in self.V[ni[0]].symmap.keys():
					newvar = None
					
					if(self.shape[node.elem.name] == "Bool"):
						newvar = (Bool('X' + str(self.number.nextn())), "Bool")
					elif(self.shape[node.elem.name] == "Int"):
						newvar = (Int('X' + str(self.number.nextn())), "Int")
					else:
						newvar = (Real('X' + str(self.number.nextn())), "Float")
					self.V[ni[0]].symmap[node.elem.name] = newvar

	def visitExprList(self, node):
		for e in node.exprlist:
			self.visit(e)

	def visitMap(self, node):
		self.visit(node.expr)
		checkPoly(self.os).visit(node.expr)
		p = self.os.visit(node.expr)
		p_poly = self.os.convertToPoly(p)
		for n in epoly.coeffs.keys():
			elist = ExprList([n, epoly.coeffs[n]])
			fcall = AST.FuncCallNode(node.func, elist)
			self.visit(fcall)

	def initV(self, v1):
		if(not "bias" in v1.symmap.keys()):
				v1.symmap["bias"] = (Real('X' + str(self.number.nextn())), "Float")
		if(not "layer" in v1.symmap.keys()):
				v1.symmap["layer"] = (Int('X' + str(self.number.nextn())), "Int")
		if(not "weight" in v1.symmap.keys()):
				v1.symmap["weight"] = [(Real('X' + str(self.number.nextn())), "Float") for i in range(self.N)]
		for s in shape.keys():
			if(not s in v1.symmap.keys()):
				if(shape[s] == "Bool"):
					newvar = (Bool('X' + str(self.number.nextn())), "Bool")
				elif(shape[s] == "Int"):
					newvar = (Int('X' + str(self.number.nextn())), "Int")
				else:
					newvar = (Real('X' + str(self.number.nextn())), "Float")
				v1.symmap[s] = newvar

	def compareV(self, v1, v2):
		self.initV(v1)
		self.initV(v2)
		newC = True
		for s in v1.symmap.keys():
			newC = And(newC, self.os.convertToZ3(v1.symmap[s]) == self.os.convertToZ3(v2.symmap[s]))

		return newC

	def visitNlistOp(self, node):
		self.visit(node.expr)
		e = self.os.visit(node.expr)

		if(node.op == "max"):
			newvar = Real('X' + str(self.number.nextn()))
			self.os.M[MAX(e)] = (newvar, "Float")
			t = False
			for i in range(len(e)):
				self.os.C.append(newvar >= self.os.convertToZ3(e[i]))
				t = Or(t, newvar == self.os.convertToZ3(e[i]))
			self.os.C.append(t)
		elif(node.op == "min"):
			newvar = Real('X' + str(self.number.nextn()))
			self.os.M[MIN(e)] = (newvar, "Float")
			t = False
			for i in range(len(e)):
				self.os.C.append(newvar <= self.os.convertToZ3(e[i]))
				t = Or(t, newvar == self.os.convertToZ3(e[i]))
			self.os.C.append(t)
		elif(node.op == "argmax"):

			newvertex = Vertex('X' + str(self.number.nextn()))
			self.os.V[newvertex.name] = newvertex
			self.os.M[ARGMAX(e, node.elem)] = (newvertex.name, "Neuron")
			t = False
			for i in range(len(e)):
				t = Or(t, compareV(self.os.V[e[i][0]], newvertex))
				self.os.C.append(self.convertToZ3(self.os.V[e[i][0]].symmap[node.elem]) <= self.convertToZ3(newvertex.symmap[node.elem]))

			self.os.C.append(t)
		elif(node.op == "argmin"):

			newvertex = Vertex('X' + str(self.number.nextn()))
			self.os.V[newvertex.name] = newvertex
			self.os.M[ARGMIN(e, node.elem)] = (newvertex.name, "Neuron")
			t = False
			for i in range(len(e)):
				t = Or(t, compareV(self.os.V[e[i][0]], newvertex))
				self.os.C.append(self.convertToZ3(self.os.V[e[i][0]].symmap[node.elem]) >= self.convertToZ3(newvertex.symmap[node.elem]))

			self.os.C.append(t)

	def visitTernary(self, node):
		self.visit(node.cond)
		self.visit(node.texpr)
		self.visit(node.fexpr)
		c = self.os.visit(node.cond)
		left = self.os.visit(node.texpr)
		right = self.os.visit(node.fexpr)
		if(isinstance(left, AND) or isinstance(left, OR) or isinstance(left, LT) or isinstance(left, GT) or
		 isinstance(left, LEQ) or isinstance(left, GEQ) or isinstance(left, EQQ) or isinstance(left, NEQ) ):
			newvar = Bool('X' + str(self.number.nextn()))
			self.os.M[Ternary(c, left, right)] = (newvar, "Bool")
			self.os.C.append(If(self.os.convertToZ3(c), self.os.convertToZ3(left) == newvar, self.os.convertToZ3(right) == newvar))
		elif(isinstance(left, tuple) and isinstance(right, tuple) and left[1] == "Neuron" and right[1] == "Neuron"):
			newvar = Vertex('X' + str(self.number.nextn()))
			V[neuron.name] = neuron
			self.os.M[Ternary(c, left, right)] = (newvar.name, "Neuron")
			self.os.C.append(If(self.os.convertToZ3(c), self.compareV(V[left[0]], newvar), self.compareV(V[right[0]], newvar)))
		else:
			newvar = Real('X' + str(self.number.nextn()))
			self.os.M[Ternary(c, left, right)] = (newvar.name, "Float")
			self.os.C.append(If(self.os.convertToZ3(c), self.os.convertToZ3(left) == newvar, self.os.convertToZ3(right) == newvar))

	def visitTraverse(self, node):
		self.visit(node.expr)
		pz3 = self.os.visit(node.p)
		Cnew = []
		vars = getVars().visit(self.constraint, self.shape)
		for v in self.V.keys():
			for var in vars.keys():
				if not var in self.V[v].symmap.keys():
					if(vars[var] == "Bool"):
						self.V[v].symmap[var] = (Bool('X' + str(self.number.nextn())), "Bool")
					elif(vars[var] == "Int"):
						self.V[v].symmap[var] = (Int('X' + str(self.number.nextn())), "Int")
					else:
						self.V[v].symmap[var] = (Real('X' + str(self.number.nextn())), "Float")

			self.store["curr'"] = (v, "Neuron")
			Cnew.append(self.os.visit(self.constraint))

		del self.store["curr'"]
		
		p = Not(Implies(And(And(self.os.C), And(Cnew)), pz3))
		s = Solver()
		s.add(p)
		if(not (s.check() == unsat)):
			raise Exception("Invarient is not true on input")

		oldV = copy.copy(self.V)
		oldM = copy.copy(self.M)
		oldC = copy.copy(self.C)
		oldStore = copy.copy(self.store)

		newvar = Real('X' + str(self.number.nextn()))
		newval = newvar
		newvar = (newvar, "Float")
		for i in range(self.N):
			neuron = Vertex('X' + str(self.number.nextn()))
			V[neuron.name] = neuron 
			const = (Real('X' + str(self.number.nextn())), "Float")
			self.visit(AST.FuncCallNode(node.stop, [(neuron.name, "Neuron"), const]))
			self.visit(AST.FuncCallNode(node.func, [(neuron.name, "Neuron"), const]))
			valf2 = self.os.visit(AST.FuncCallNode(node.stop, [(neuron.name, "Neuron"), const]))
			valf3 = self.os.visit(AST.FuncCallNode(node.func, [(neuron.name, "Neuron"), const]))
			newvar = Add(newvar, Mult(const, (neuron, "Neuron")))
			newval = newval + If(self.os.convertToZ3(valf2), self.os.convertToZ3(valf3), self.os.convertToZ3(Mult(const, (neuron.name, "Neuron"))))

		s = Solver()
		newstore1 = self.store 
		newstore1[node.expr.name] = newvar 
		self.os.store = newstore1
		p1 = self.os.visit(node.p)

		temp_var = Real('X' + str(self.number.nextn()))
		self.C.append(temp_var == newval)
		newstore2 = self.store 
		newstore2[node.expr.name] = (temp_var, "Float") 
		self.os.store = newstore2
		p2 = self.os.visit(node.p)
		Cnew = []

		for v in self.V.keys():
			for var in vars.keys():
				if not var in V[v].symmap.keys():
					if(vars[var] == "Bool"):
						V[v].symmap[var] = (Bool('X' + str(self.number.nextn())), "Bool")
					elif(vars[var] == "Int"):
						V[v].symmap[var] = (Int('X' + str(self.number.nextn())), "Int")
					else:
						V[v].symmap[var] = (Real('X' + str(self.number.nextn())), "Float")

			self.store["curr'"] = (v, "Neuron")
			Cnew.append(self.os.visit(self.constraint))

		del self.store["curr'"]

		p = Not(Implies(And(And(And(self.C), And(Cnew)), p1), p2))
		s.add(p)
		if(not (s.check() == unsat)):
			raise Exception("Induction step is not true")
		self.M = oldM
		self.V = oldV
		self.store = oldStore
		self.os.store = oldStore
		self.C = oldC 
		self.C.append(p2)

	def visitSub(self, node):
		pass

	def visitTransRetBasic(self, node):
		self.visit(node.exprlist)

	def visitTransRetIf(self, node):
		self.visit(node.cond)
		self.visit(node.tret)
		self.visit(node.fret)


class checkPoly(astVisitor.ASTVisitor):

	def __init__(self, os):
		self.os = os

	def visitExprList(self, node: AST.ExprListNode):
		for e in node.exprlist:
			self.visit(e)
			
	def visitBinOp(self, node: AST.BinOpNode):
		self.visit(node.left)
		self.visit(node.right)
		
	def visitUnOp(self, node: AST.UnOpNode):
		self.visit(node.expr)

	def visitNlistOp(self, node: AST.NlistOpNode):
		self.visit(node.expr)

	def visitVar(self, node: AST.VarNode):
		pass

	def visitNeuron(self, node: AST.NeuronNode):
		pass

	def visitInt(self, node: AST.ConstIntNode):
		pass

	def visitFloat(self, node: AST.ConstFloatNode):
		pass

	def visitBool(self, node: AST.ConstBoolNode):
		pass

	def visitCurr(self, node: AST.CurrNode):
		pass

	def visitPrev(self, node: AST.PrevNode):
		pass

	def visitEpsilon(self, node: AST.EpsilonNode):
		pass

	def visitTernary(self, node: AST.TernaryNode):
		self.visit(node.cond)
		self.visit(node.texpr)
		self.visit(node.fexpr)


	def visitTraverse(self, node: AST.TraverseNode):
		self.visit(node.expr)
		self.visit(node.priority)
		self.visit(node.stop)
		self.visit(node.func)


	def visitSum(self, node: AST.SumNode):
		self.visit(node.expr)

	def visitSub(self, node: AST.SubNode):
		self.visit(node.listexpr)

	def visitMap(self, node: AST.MapNode):
		self.visit(node.expr)

	def visitDot(self, node: AST.DotNode):
		self.visit(node.left)
		self.visit(node.right)

	
	def visitFuncCall(self, node: AST.FuncCallNode):
		name = node.name.name
		self.visit(F[name].expr)


	def visitSeq(self, node: AST.SeqNode):
		self.visit(node.stmt1)
		self.visit(node.stmt2)

	def visitFlow(self, node):
		pass

	def visitProg(self, node: AST.ProgramNode):
		self.visit(node.shape)
		self.visit(node.stmt)

	def visitGetMetadata(self, node: AST.GetMetadataNode):
		pass

	def visitGetElement(self, node):
		n = self.os.visit(node.expr)
		if(isinstance(self.os.V[n[0]].symmap[n.elem.name], tuple)):
			if(self.os.V[n[0]].symmap[n.elem.name][1] == "Float"):
				oldvar = self.os.V[n[0]].symmap[n.elem.name][0]
				newvar = (Real(n.elem.name + '_X' + str(self.number.nextn())), "Float")

				for i in range(self.N):
					neuron = Vertex('X' + str(self.number.nextn()))
					V[neuron.name] = neuron 
					const = (Real('X' + str(self.number.nextn())), "Float")
					newvar = Add(newvar, Mult(const, (neuron, "Neuron")))

				self.os.V[n[0]].symmap[n.elem.name] = newvar
				self.os.C.append(oldvar == self.os.convertToZ3(newvar))

class getVars(astVisitor.ASTVisitor):

	def __init__(self, constraint, shape):
		self.constraint = constraint
		self.shape = shape
		self.vars = {}


	def visitExprList(self, node: AST.ExprListNode):
		for e in node.exprlist:
			self.visit(e)
			
	def visitBinOp(self, node: AST.BinOpNode):
		self.visit(node.left)
		self.visit(node.right)
		
	def visitUnOp(self, node: AST.UnOpNode):
		self.visit(node.expr)

	def visitNlistOp(self, node: AST.NlistOpNode):
		self.visit(node.expr)

	def visitVar(self, node: AST.VarNode):
		pass

	def visitNeuron(self, node: AST.NeuronNode):
		pass

	def visitInt(self, node: AST.ConstIntNode):
		pass

	def visitFloat(self, node: AST.ConstFloatNode):
		pass

	def visitBool(self, node: AST.ConstBoolNode):
		pass

	def visitCurr(self, node: AST.CurrNode):
		pass

	def visitPrev(self, node: AST.PrevNode):
		pass

	def visitEpsilon(self, node: AST.EpsilonNode):
		pass

	def visitTernary(self, node: AST.TernaryNode):
		self.visit(node.cond)
		self.visit(node.texpr)
		self.visit(node.fexpr)


	def visitTraverse(self, node: AST.TraverseNode):
		self.visit(node.expr)
		self.visit(node.priority)
		self.visit(node.stop)
		self.visit(node.func)


	def visitSum(self, node: AST.SumNode):
		self.visit(node.expr)

	def visitSub(self, node: AST.SubNode):
		self.visit(node.listexpr)

	def visitMap(self, node: AST.MapNode):
		self.visit(node.expr)

	def visitDot(self, node: AST.DotNode):
		self.visit(node.left)
		self.visit(node.right)

	
	def visitFuncCall(self, node: AST.FuncCallNode):
		name = node.name.name
		self.visit(F[name].expr)


	def visitSeq(self, node: AST.SeqNode):
		self.visit(node.stmt1)
		self.visit(node.stmt2)

	def visitFlow(self, node):
		pass

	def visitProg(self, node: AST.ProgramNode):
		self.visit(node.shape)
		self.visit(node.stmt)

	def visitGetMetadata(self, node: AST.GetMetadataNode):
		pass

	def visitGetElement(self, node):
		self.vars[node.elem.name] = self.shape[node.elem.name]

	def visitPropTermBasic(self, node: AST.PropTermBasicNode):
		self.visit(node.term)

	def visitPropTermIn(self, node: AST.PropTermInNode):
		self.visit(node.n)
		self.visit(node.z)

	def visitPropTermOp(self, node: AST.PropTermOpNode):
		self.visit(node.leftpt)
		self.visit(node.rightpt)
		

	def visitSingleProp(self, node: AST.SinglePropNode):
		self.visit(node.leftpt)
		self.visit(node.rightpt)

	def visitDoubleProp(self, node: AST.DoublePropNode):
		self.visit(node.leftprop)
		self.visit(node.rightprop)


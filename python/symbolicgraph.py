import astVisitor
import ast as AST
from z3 import *
from value import *
from verify import *
import copy


class SymbolicGraph(astVisitor):

	def __init__(self, store, F):
		self.M = {}
		self.V = {}
		self.C = []
		self.store = store
		self.F = F
		self.shape = []
		self.os = Evaluate(self.store, self.F, self.M, self.V, self.C, self.shape)
		self.number = Number()
		self.N = 3

	def vistInt(self, node):
		pass

	def visitVar(self, node):
		pass

	def visitFloat(self, node):
		pass

	def visitBool(self, node):
		pass

	def visitBinop(self, node):
		self.visit(left)
		self.visit(right)

	def visitUnOp(self, node):
		self.visit(node.expr)

	def visitSum(self, node):
		self.visit(node.expr)

	def visitDot(self, node):
		self.visit(left)
		self.visit(right)

	def visitGetMetadata(self, node):
		self.visit(node.expr)
		n = self.os.visit(node.expr)
		if not node.metadata in self.V[n].symmap.keys():
			newvar = None
			if(node.metadata == "bias"):
				newvar = (Real('X' + self.number.nextn()), "Bool")
			elif(node.metadate == "weight"):
				newvar = [(Real('X' + self.number.nextn()), "Float") for i in self.N]
			elif(node.metadata == "layer"):
				newvar = (Int('X' + self.number.nextn()), "Int")
			self.V[n].symmap[node.metadata] = newvar

	def visitGetElement(self, node):
		self.visit(node.expr)
		n = self.os.visit(node.expr)
		if not node.elem in self.V[n].symmap.keys():
			newvar = None
			if(shape[node.elem] == "Bool"):
				newvar = (Bool('X' + self.number.nextn()), "Bool")
			elif(shape[node.elem] == "Int"):
				newvar = (Int('X' + self.number.nextn()), "Int")
			else:
				newvar = (Real('X' + self.number.nextn()), "Float")
			self.V[n].symmap[node.elem] = newvar

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
				v1.symmap["bias"] = (Real('X' + self.number.nextn()), "Float")
		if(not "layer" in v1.symmap.keys()):
				v1.symmap["layer"] = (Int('X' + self.number.nextn()), "Int")
		if(not "weight" in v1.symmap.keys()):
				v1.symmap["weight"] = [(Real('X' + self.number.nextn()), "Float") for i in range(self.N)]
		for s in shape.keys():
			if(not s in v1.symmap.keys()):
				if(shape[s] == "Bool"):
					newvar = (Bool('X' + self.number.nextn()), "Bool")
				elif(shape[s] == "Int"):
					newvar = (Int('X' + self.number.nextn()), "Int")
				else:
					newvar = (Real('X' + self.number.nextn()), "Float")
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
			newvar = Real('X' + self.number.nextn())
			self.os.M[MAX(e)] = (newvar, "Float")
			t = False
			for i in range(len(e)):
				self.os.C.append(newvar >= self.os.convertToZ3(e[i]))
				t = Or(t, newvar == self.os.convertToZ3(e[i]))
			self.os.C.append(t)
		elif(node.op == "min"):
			newvar = Real('X' + self.number.nextn())
			self.os.M[MIN(e)] = (newvar, "Float")
			t = False
			for i in range(len(e)):
				self.os.C.append(newvar <= self.os.convertToZ3(e[i]))
				t = Or(t, newvar == self.os.convertToZ3(e[i]))
			self.os.C.append(t)
		elif(node.op == "argmax"):

			newvertex = Vertex('X' + self.number.nextn())
			self.os.V[newvertex.name] = newvertex
			self.os.M[ARGMAX(e, node.elem)] = (newvertex.name, "Neuron")
			t = False
			for i in range(len(e)):
				t = Or(t, compareV(self.os.V[e[i][0]], newvertex))
				self.os.C.append(self.convertToZ3(self.os.V[e[i][0]].symmap[node.elem]) <= self.convertToZ3(newvertex.symmap[node.elem]))

			self.os.C.append(t)
		elif(node.op == "argmin"):

			newvertex = Vertex('X' + self.number.nextn())
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
			if(isinstance(left, AND) or isinstance(left, OR) or isinstance(left, LT) or isinstance(left, GT) 
			 isinstance(left, LEQ) or isinstance(left, GEQ) or isinstance(left, EQQ) or isinstance(left, NEQ) ):
				newvar = Bool('X' + self.number.nextn())
				self.os.M[Ternary(c, left, right)] = (newvar, "Bool")
				self.os.C.append(If(self.os.convertToZ3(c), self.os.convertToZ3(left) == newvar, self.os.convertToZ3(right) == newvar))
			elif(isinstance(left, tuple) and isinstance(right, tuple) and left[1] == "Neuron" and right[1] == "Neuron"):
				newvar = Vertex('X' + self.number.nextn())
				V[neuron.name] = neuron
				self.os.M[Ternary(c, left, right)] = (newvar.name, "Neuron")
				self.os.C.append(If(self.os.convertToZ3(c), self.compareV(V[left[0]], newvar), self.compareV(V[right[0]], newvar)))
			else:
				newvar = Real('X' + self.number.nextn())
				self.os.M[Ternary(c, left, right)] = (newvar.name, "Float")
				self.os.C.append(If(self.os.convertToZ3(c), self.os.convertToZ3(left) == newvar, self.os.convertToZ3(right) == newvar))

		def visitTraverse(self, node):
			self.visit(node.expr)
			pz3 = self.os.visit(node.p)
			p = Not(Implies(And(self.os.C), pz3))
			s = Solver()
			s.add(p)
			if(not (s.check() == unsat)):
				raise Exception("Invarient is not true on input")

			oldV = copy.copy(self.V)
			oldM = copy.copy(self.M)
			oldC = copy.copy(self.C)
			oldStore = copy.copy(self.store)

			newvar = Real('X' + self.number.nextn())
			newval = newvar
			newvar = (newvar, "Float")
			for i in range(self.N):
				neuron = Vertex('X' + self.number.nextn())
				V[neuron.name] = neuron 
				const = (Real('X' + self.number.nextn()), "Float")
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

			temp_var = Real('X' + self.number.nextn())
			self.C.append(temp_var == newval)
			newstore2 = self.store 
			newstore2[node.expr.name] = (temp_var, "Float") 
			self.os.store = newstore2
			p2 = self.os.visit(node.p)
			p = Not(Implies(And(And(self.C), p1), p2))
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



class checkPoly(astVisitor):

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
		if(isinstance(self.os.V[n[0]].symmap[n.elem], tuple):
			if(self.os.V[n[0]].symmap[n.elem][1] == "Float"):
				oldvar = self.os.V[n[0]].symmap[n.elem][0]
				newvar = (Real('X' + self.number.nextn()), "Float")

				for i in range(self.N):
					neuron = Vertex('X' + self.number.nextn())
					V[neuron.name] = neuron 
					const = (Real('X' + self.number.nextn()), "Float")
					newvar = Add(newvar, Mult(const, (neuron, "Neuron")))

				self.os.V[n[0]].symmap[n.elem] = newvar
				self.os.C.append(oldvar == self.os.convertToZ3(newvar))

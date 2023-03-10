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

	def visitBaseType(self, node: AST.BaseTypeNode):
		return node.name

	def visitArrayType(self, node: AST.ArrayTypeNode):
		return ArrayType(node.base)

	def visitArgList(self, node: AST.ArgListNode):
		return [t[0].name for t in node.arglist]

	def visitExprList(self, node: AST.ExprListNode):
		listtype = []
		for e in node.exprlist:
			listtype.append(self.visit(e))

		return listtype

	def BinOphelper(self, node, ltype, rtype):
		accepted = ["Int", "Float", "Neuron", "PolyExp", "ZonoExp"]
		if(node.op == "+" or node.op == "-"):
			if(ltype in accepted and rtype in accepted):
				if(self.comparable(ltype, rtype)):
					return self.comparable(ltype, rtype)
				else:
					raise TypeMismatchException(str(ltype) + " and " + str(rtype) + " are not comparable")
			else:
				raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))

		elif(node.op == "*"):
			if(self.isType(ltype, "Float")):
				if(rtype in accepted):
					return self.comparable(ltype, rtype)
			elif(self.isType(rtype, "Float")):
				if(ltype in accepted):
					return self.comparable(rtype, ltype)
			raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))

		elif(node.op == "/"):
			if(self.isType(ltype, "Float") and self.isType(rtype, "Float")):
				return self.comparable(ltype, rtype)
			elif(self.isType(rtype, "Float")):
				if(ltype == "PolyExp" or ltype == "ZonoExp"):
					return ltype

		elif(node.op == "and" or node.op == "or"):
			if(ltype == "Bool" and rtype =="Bool"):
				return "Bool"
			else:
				raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))
		else: #<=, >=, ==
			if(self.isType(ltype, "Float") and self.isType(rtype, "Float")):
				return "Bool"
			else:
				raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))


	def visitBinOp(self, node: AST.BinOpNode):
		ltype = self.visit(node.left)
		rtype = self.visit(node.right)

		#Don't need the if statement before calling the function unless you define some operations on lists
		if(not isinstance(ltype, ArrayType) and not isinstance(rtype, ArrayType)):
			return self.BinOphelper(node, ltype, rtype)
		else:
			raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))

		
	def visitUnOp(self, node: AST.UnOpNode):
		t = self.visit(node.expr)
		if(node.op == "-"):
			if(t == "Int" or t =="Float"):
				return t
			else:
				raise TypeMismatchException("- not defined on " + str(t))
		elif(node.op == "~"):
			if(t == "Bool"):
				return t
			else:
				raise TypeMismatchException("~ not defined on " + str(t))
		else:
			assert False

	def visitNlistOp(self, node: AST.NlistOpNode):
		exptype = self.visit(node.expr)
		if(not isinstance(exptype, ArrayType)):
			raise TypeMismatchException(node.op + " requires Neuron list as first argument")
		elif(not exptype.base == "Neuron"):
			raise TypeMismatchException(node.op + " requires Neuron list as first argument")

		if(not node.elem.name in [x[1] for x in self.shape]):
			raise TypeMismatchException(node.elem.name + " not found in shape declaration")
		
		if(node.op == "min" or node.op == "max"):
			return self.vars[node.elem.name]
		elif(node.op == "argmin" or node.op == "argmax"):
			return "Neuron"
		else:
			assert False

	def visitVar(self, node: AST.VarNode):
		if not node.name in self.vars.keys():
			raise UndefinedVarException(node.name + " is undefined")
		else:
			return self.vars[node.name]

	def visitNeuron(self, node: AST.NeuronNode):
		return "Neuron"

	def visitInt(self, node: AST.ConstIntNode):
		return "Int"

	def visitFloat(self, node: AST.ConstFloatNode):
		return "Float"

	def visitBool(self, node: AST.ConstBoolNode):
		return "Bool"

	def visitCurr(self, node: AST.CurrNode):
		if(self.currdefined):
			return "Neuron"
		else:
			raise UndefinedVarException("Curr is not defined")

	def visitPrev(self, node: AST.PrevNode):
		if(self.currdefined):
			return ArrayType("Neuron")
		else:
			raise UndefinedVarException("Prev is not defined")

	def visitEpsilon(self, node: AST.EpsilonNode):
		return "Noise"

	def visitTernary(self, node: AST.TernaryNode):
		ctype = self.visit(node.cond)
		if(not self.isType(ctype, "Bool")):
			raise TypeMismatchException("Condition has to be boolean")

		ltype = self.visit(node.texpr)
		rtype = self.visit(node.fexpr)
		if(self.comparable(ltype, rtype)):
			return self.comparable(ltype, rtype)
		else:
			raise TypeMismatchException(str(ltype) + " and " + str(rtype) + " are not comparable")

	def visitGetMetadata(self, node: AST.GetMetadataNode):
		etype = self.visit(node.expr)
		if(etype == "Neuron"):
			if(node.metadata.name == "layer"):
				return "Int"
			elif(node.metadata.name == "bias"):
				return "Float"
			elif(node.metadata.name == "weight"):
				return ArrayType("Float")
			else:
				assert False
		elif(isinstance(etype, ArrayType) and etype.base == "Neuron"):
			if(node.metadata.name == "layer"):
				return ArrayType("Int")
			elif(node.metdata.name == "bias"):
				return ArrayType("Float")
			elif(node.metadata.name == "weight"):
				return ArrayType(ArrayType("Float"))
			else:
				assert False
		else:
			raise TypeMismatchException("To access metadata, expression must be Neuron or Neuron List")

	def visitGetElement(self, node: AST.GetElementNode):
		etype = self.visit(node.expr)
		if(not node.elem.name in self.vars.keys()):
			raise UndefinedVarException(node.elem.name + " is undefined")

		elemtype = self.vars[node.elem.name]
		if(not (elemtype, node.elem.name) in self.shape):
			raise TypeMismatchException(node.elem.name + " is not part of the shape")

		if(etype == "Neuron"):
			return elemtype
		elif(isinstance(etype, ArrayType) and etype.base == "Neuron"):
			return ArrayType(elemtype)
		else:
			raise TypeMismatchException("To access shape variables, expression must be Neuron or Neuron List")


	def visitTraverse(self, node: AST.TraverseNode):
		exprtype = self.visit(node.expr)
		if(not exprtype == "PolyExp"):
			raise TypeMismatchException("Traverse must be called on PolyExp")

		prtype = self.visit(node.priority)
		sttype = self.visit(node.stop)
		if(not self.isType(prtype, ArrowType("Neuron", "Float"))):
			raise TypeMismatchException("Traverse priority function is wrong type")
		elif(not self.isType(sttype, ArrowType("Neuron", "Bool"))):
			raise TypeMismatchException("Traverse stopping function is wrong type")

		ftype = self.visit(node.func)
		if(not isinstance(ftype, ArrowType)):
			raise TypeMismatchException("Last argument to Traverse must be a function")
		left = ftype.tleft
		right = ftype.tright
		if(isinstance(left, list)):
			if(len(left) == 2):
				if(left[0] == "Neuron" and left[1] == "Float"):
					if(self.isType(right, "PolyExp")):
						self.visit(node.p)
						if(right == "Neuron"):
							return "PolyExp"
						else:
							return right

		raise TypeMismatchException("Last argument to traverse is the wrong type")


	def visitSum(self, node: AST.SumNode):
		exptype = self.visit(node.expr)

		if(not isinstance(exptype, ArrayType)):
			raise TypeMismatchException("Expression passed to sum must be a list")

		base = exptype.base
		if(base == "Int" or base == "Float" or base == "PolyExp" or base == "ZonoExp"):
			return base
		elif (base == "Neuron"):
			return "PolyExp"
		else:
			raise TypeMismatchException(base + " list cannot be passed to sum")

	def visitSub(self, node: AST.SubNode):
		listtype = self.visit(node.listexpr)
		if(isinstance(listtype, ArrayType)):
			etype = self.visit(node.expr)
			if(self.isType(etype, listtype.base)):
				return listtype
			else:
				raise TypeMismatchException("Second argument of Sub should be of the same type as values in the list")
		else:
			raise TypeMismatchException("First argument of Sub should be a list")

	def visitMap(self, node: AST.MapNode):
		exptype = self.visit(node.expr)
		if(not exptype == "PolyExp"):
			raise TypeMismatchException("Left of Map function must be a PolyExp")

		ftype = self.visit(node.func)
		if(not isinstance(ftype, ArrowType)):
			raise TypeMismatchException("Argument to Map must be a function")
		left = ftype.tleft
		right = ftype.tright
		if(isinstance(left, list)):
			if(len(left) == 2):
				if(left[0] == "Neuron" and left[1] == "Float"):
					if(self.isType(right, "PolyExp")):
						if(right == "Neuron"):
							return "PolyExp"
						else:
							return right

		raise TypeMismatchException("Function argument to Map is the wrong type")


	def visitDot(self, node: AST.DotNode):
		tleft = self.visit(node.left)
		tright = self.visit(node.right)

		if(not isinstance(tleft, ArrayType)):
			raise TypeMismatchException("Left of dot product has to be Neuron list")
		elif(not tleft.base == "Neuron"):
			raise TypeMismatchException("Left of dot product has to be Neuron list")

		if(self.isType(tright, ArrayType("Float"))):
			return "PolyExp"
		else:
			raise TypeMismatchException("Right of dot product has to be Int or Float List")

	def visitFuncCall(self, node: AST.FuncCallNode):
		name = node.name.name
		


	def visitSeq(self, node: AST.SeqNode):
		self.visit(node.stmt1)
		self.visit(node.stmt2)

	def visitProg(self, node: AST.ProgramNode):
		self.visit(node.shape)
		self.visit(node.stmt)

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

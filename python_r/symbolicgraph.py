import astVisitor
import ast as AST
from z3 import *
from value import *
from symbolicos import *
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
		self.os = SymbolicOperationalSemantics(self.store, self.F, self.M, self.V, self.C, self.shape)
		self.number = Number()
		self.N = 3
		self.currop = None #Stores the relationship between curr and prev for traverse proof

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

	def visitListOp(self, node):
		self.visit(node.expr)

	# def getMember(self, l, i):
	# 	member = Real('m' + str(self.number.nextn()))
	# 	conds = []
	# 	for j in len(l):
	# 		conds.append(Implies(i==j, member==l[j]))
	# 	return member, conds

	# def getLength(self, l, f):
	# 	length = 0
	# 	for i in range(len(l)):
	# 		length += If(f(i, l[i]), 1, 0)
	# 	return length 

	def visitDot(self, node):
		self.visit(node.left)
		self.visit(node.right)

	def get_getMetadata(self, n, node):
		if(isinstance(n, IF)):
			self.get_getMetadata(n.left, node)
			self.get_getMetadata(n.right, node)
		else:
			if(not isinstance(n, list)):
				if not node.metadata.name in self.V[n[0]].symmap.keys():
					newvar = None
					if(node.metadata.name == "bias"):
						newvar = (Real('X' + str(self.number.nextn())), "Float")
					elif(node.metadata.name == "weight"):
						newvar_list = [None]*(self.N)
						for i in range(self.N):
							newvar_list[i] = (Real('X' + str(self.number.nextn())), "Float")
						newvar = newvar_list
					elif(node.metadata.name == "layer"):
						newvar = (Int('X' + str(self.number.nextn())), "Int")
					elif(node.metadata.name == "serial"):
						newvar = (Int('X' + str(self.number.nextn())), "Int")
					elif(node.metadata.name == "local_serial"):
						newvar = (Int('X' + str(self.number.nextn())), "Int")
					self.V[n[0]].symmap[node.metadata.name] = newvar
			else:
				for ni in n:
					self.get_getMetadata(ni, node)

	def visitGetMetadata(self, node):
		self.visit(node.expr)
		n = self.os.visit(node.expr)
		self.get_getMetadata(n, node)


	def get_getElement(self, n, node):
		if(isinstance(n, IF)):
			self.get_getElement(n.left, node)
			self.get_getElement(n.right, node)
		else:
			if(not isinstance(n, list)):
				if not node.elem.name in self.V[n[0]].symmap.keys():
					newvar = None
					if(self.shape[node.elem.name] == "Bool"):
						newvar = (Bool(str(node.elem.name) + '_X' + str(self.number.nextn())), self.shape[node.elem.name])
					elif(self.shape[node.elem.name] == "Int"):
						newvar = (Int(str(node.elem.name) + '_X' + str(self.number.nextn())), self.shape[node.elem.name])
					else:
						newvar = (Real(str(node.elem.name) + '_X' + str(self.number.nextn())), self.shape[node.elem.name])
					self.V[n[0]].symmap[node.elem.name] = newvar
			else:
				for ni in n:
					self.get_getElement(ni, node)
				

	def visitGetElement(self, node):
		self.visit(node.expr)
		n = self.os.visit(node.expr)
		self.get_getElement(n, node)

	def visitExprList(self, node):
		for e in node.exprlist:
			self.visit(e)

	def get_map(self, val, node):
		if isinstance(val, IF):
			self.get_map(val.left, node)
			self.get_map(val.right, node)
		else:
			if self.os.get_type(val)=='PolyExp':
				expr = self.os.convertToPoly(val)
			else:
				expr = self.os.convertToZono(val)
			for n in expr.coeffs.keys():
				if isinstance(node.func, AST.VarNode):
					elist = []
				else:	
					elist = self.os.visit(node.func)
				elist = AST.ExprListNode(elist + [n, expr.coeffs[n]])
				fcall = AST.FuncCallNode(node.func, elist)
				self.visitFuncCall(fcall, True)

	def visitMap(self, node):
		self.visit(node.expr)
		checkPoly(self.os).visit(node.expr)
		p = self.os.visit(node.expr)
		self.get_map(p, node)
		# p_poly = self.os.convertToPoly(p)
		# for n in p_poly.coeffs.keys():
		# 	if isinstance(node.func, AST.VarNode):
		# 		elist = []
		# 	else:	
		# 		elist = self.os.visit(node.func)
		# 	elist = AST.ExprListNode(elist + [n, p_poly.coeffs[n]])
		# 	fcall = AST.FuncCallNode(node.func, elist)
		# 	self.visitFuncCall(fcall, True)

	def visitFuncCall(self, node, preeval = False):
		func = self.F[node.name.name]

		newvars = []
		oldvalues = {}

		if(not preeval):
			self.visit(node.arglist)
			elist = self.os.visit(node.arglist)
		else:
			elist = node.arglist.exprlist

		for (exp,(t, arg)) in zip(elist, func.decl.arglist.arglist):
			if arg.name in self.store.keys():
				oldvalues[arg.name] = self.store[arg.name]
			else:
				newvars.append(arg.name)

			self.store[arg.name] = exp

		self.visit(func.expr)
		
		for v in newvars:
			del self.store[v]

		for ov in oldvalues.keys():
			self.store[ov] = oldvalues[ov]


	# def initV(self, v1):
	# 	if(not "bias" in v1.symmap.keys()):
	# 		v1.symmap["bias"] = (Real('X' + str(self.number.nextn())), "Float")
	# 	if(not "layer" in v1.symmap.keys()):
	# 		v1.symmap["layer"] = (Int('X' + str(self.number.nextn())), "Int")
	# 	if(not "weight" in v1.symmap.keys()):
	# 		v1.symmap["weight"] = [(Real('X' + str(self.number.nextn())), "Float") for i in range(self.N)]
	# 	if(not "local_serial" in v1.symmap.keys()):
	# 		v1.symmap["local_serial"] = (Int('X' + str(self.number.nextn())), "Int")
	# 	if(not "serial" in v1.symmap.keys()):
	# 		v1.symmap["serial"] = (Int('X' + str(self.number.nextn())), "Int")

	# 	for s in self.shape.keys():
	# 		if(not s in v1.symmap.keys()):
	# 			if(self.shape[s] == "Bool"):
	# 				newvar = (Bool('X' + str(self.number.nextn())), "Bool")
	# 			elif(self.shape[s] == "Int"):
	# 				newvar = (Int('X' + str(self.number.nextn())), "Int")
	# 			else:
	# 				newvar = (Real('X' + str(self.number.nextn())), "Float")
	# 			v1.symmap[s] = newvar

	# def compareV(self, v1, v2):
	# 	self.initV(v1)
	# 	self.initV(v2)
	# 	newC = True
	# 	for s in v1.symmap.keys():
	# 		newC = And(newC, self.os.convertToZ3(v1.symmap[s]) == self.os.convertToZ3(v2.symmap[s]))

	# 	return newC

	def visitArgmaxOp(self, node):
		self.visit(node.expr)

	def visitMaxOp(self, node):
		self.visit(node.expr1)
		self.visit(node.expr2)

	def visitMaxOpList(self, node):
		self.visit(node.expr)

	def visitTernary(self, node):
		self.visit(node.cond)
		self.visit(node.texpr)
		self.visit(node.fexpr)

	def visitTraverse(self, node):
		self.visit(node.expr)
		Cnew = []
		gv = getVars(self.constraint, self.shape)
		gv.visit(self.constraint)
		vars = gv.vars

		for v in self.V.keys():
			for var in vars.keys():
				if not var in self.V[v].symmap.keys():
					if(vars[var] == "Bool"):
						self.V[v].symmap[var] = (Bool('X' + str(self.number.nextn())), "Bool")
					elif(vars[var] == "Int"):
						self.V[v].symmap[var] = (Int('X' + str(self.number.nextn())), "Int")
					else:
						self.V[v].symmap[var] = (Real('X' + str(self.number.nextn())), "Float")

			self.store["curr_new"] = (v, "Neuron")
			Cnew.append(self.os.visit(self.constraint))

		del self.store["curr_new"]

		Cnew.append(self.currop) #ADD definition of current related to prev.

		e = self.os.visit(node.expr)
		self.store["curr_new"] = e
		pz3 = self.os.visit(node.p)
		del self.store["curr_new"]
		
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
			self.V[neuron.name] = neuron 
			const = (Real('X' + str(self.number.nextn())), "Float")
			if(isinstance(node.stop, AST.VarNode)):
				self.visitFuncCall(AST.FuncCallNode(node.stop, AST.ExprListNode([(neuron.name, "Neuron"), const])), True)
			if(isinstance(node.func, AST.VarNode)):
				self.visitFuncCall(AST.FuncCallNode(node.func, AST.ExprListNode([(neuron.name, "Neuron"), const])), True)

			if(isinstance(node.stop, AST.VarNode)):
				valf2 = self.os.visitFuncCall(AST.FuncCallNode(node.stop, AST.ExprListNode([(neuron.name, "Neuron"), const])), True)
			else: #expression like True
				valf2 = self.os.visit(node.stop)

			if(isinstance(node.func, AST.VarNode)):
				valf3 = self.os.visitFuncCall(AST.FuncCallNode(node.func, AST.ExprListNode([(neuron.name, "Neuron"), const])), True)
			else: #expression like True
				valf3 = self.os.visit(node.func)

			newvar = ADD(newvar, MULT(const, (neuron.name, "Neuron")))
			newval = newval + If(self.os.convertToZ3(valf2), self.os.convertToZ3(valf3), self.os.convertToZ3(MULT(const, (neuron.name, "Neuron"))))

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
				if not var in self.V[v].symmap.keys():
					if(vars[var] == "Bool"):
						self.V[v].symmap[var] = (Bool('X' + str(self.number.nextn())), "Bool")
					elif(vars[var] == "Int"):
						self.V[v].symmap[var] = (Int('X' + str(self.number.nextn())), "Int")
					else:
						self.V[v].symmap[var] = (Real('X' + str(self.number.nextn())), "Float")

			self.store["curr_new"] = (v, "Neuron")
			Cnew.append(self.os.visit(self.constraint))

		del self.store["curr_new"]

		p = Not(Implies(And(And(And(self.C), And(Cnew)), p1), p2))
		s.add(p)
		if(not (s.check() == unsat)):
			raise Exception("Induction step is not true")
		self.os.M = oldM
		self.M = oldM
		self.os.V = oldV
		self.V = oldV
		self.store = oldStore
		self.os.store = oldStore
		self.C = oldC
		self.os.C = oldC
		self.os.C.append(p2)

		if(isinstance(node.priority, AST.VarNode)):
			p_name = node.priority.name
		else:
			p_name = self.os.visit(node.priority)

		if(isinstance(node.stop, AST.VarNode)):
			s_name = node.stop.name
		else:
			s_name = self.os.visit(node.stop)

		self.os.M[TRAVERSE(e, node.direction, p_name, s_name, node.func.name)] = (temp_var, "Float") 

	# def visitTransRetBasic(self, node):
	# 	self.visit(node.exprlist)

	# def visitTransRetIf(self, node):
	# 	self.visit(node.cond)
	# 	self.visit(node.tret)
	# 	self.visit(node.fret)


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

	def visitArgmaxOp(self, node: AST.ArgmaxOpNode):
		self.visit(node.expr)
		self.visit(node.func)

	def visitMaxOp(self, node: AST.MaxOpNode):
		self.visit(node.expr)

	def visitMaxOpList(self, node: AST.MaxOpListNode):
		self.visit(node.expr1)
		self.visit(node.expr2)

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


	def visitListOp(self, node: AST.ListOpNode):
		self.visit(node.expr)

	def visitMap(self, node: AST.MapNode):
		self.visit(node.expr)

	def visitDot(self, node: AST.DotNode):
		self.visit(node.left)
		self.visit(node.right)

	
	def visitFuncCall(self, node: AST.FuncCallNode):
		name = node.name.name
		self.visit(F[name].expr)
		self.visit(node.arglist)


	# def visitSeq(self, node: AST.SeqNode):
	# 	self.visit(node.stmt1)
	# 	self.visit(node.stmt2)

	# def visitFlow(self, node):
	# 	pass

	# def visitProg(self, node: AST.ProgramNode):
	# 	self.visit(node.shape)
	# 	self.visit(node.stmt)

	def visitGetMetadata(self, node: AST.GetMetadataNode):
		self.visit(node.expr)

	def get_getElement(self, val, node):
		if isinstance(val, list):
			for v in val:
				self.get_getElement(v, node)
		elif isinstance(val, IF):
			self.get_getElement(val.left, node)
			self.get_getElement(val.right, node)
		else:
			name = node.elem.name
			if isinstance(self.os.V[val[0]].symmap[name], tuple):
				if (self.os.V[val[0]].symmap[name])[1] == 'PolyExp':
					oldvar = self.os.V[val[0]].symmap[name][0]
					newvar = (Real('X' + str(self.number.nextn())), "Float")

					for i in range(self.N):
						neuron = Vertex('X' + str(self.number.nextn()))
						V[neuron.name] = neuron 
						const = (Real('X' + str(self.number.nextn())), "Float")
						newvar = ADD(newvar, MULT(const, (neuron, "Neuron")))

					self.os.V[val[0]].symmap[name] = newvar
					self.os.C.append(oldvar == self.os.convertToZ3(newvar))
				elif (self.os.V[val[0]].symmap[name])[1] == 'ZonoExp':
					oldvar = self.os.V[val[0]].symmap[name][0]
					newvar = (Real('X' + str(self.number.nextn())), "Float")

					for i in range(self.N):
						epsilon = Real('eps' + str(self.number.nextn()))
						const = (Real('X' + str(self.number.nextn())), "Float")
						newvar = ADD(newvar, MULT(const, (epsilon, "Noise")))
						self.os.C.append(epsilon <= 1)
						self.os.C.append(epsilon >= -1)

					self.os.V[val[0]].symmap[name] = newvar
					self.os.C.append(oldvar == self.os.convertToZ3(newvar))
			return

	def visitGetElement(self, node):
		n = self.os.visit(node.expr)
		get_getElement(n, node)
		# if(isinstance(self.os.V[n[0]].symmap[n.elem.name], tuple)):
		# 	if(self.os.V[n[0]].symmap[n.elem.name][1] == "Float"):
		# 		oldvar = self.os.V[n[0]].symmap[n.elem.name][0]
		# 		newvar = (Real('X' + str(self.number.nextn())), "Float")

		# 		for i in range(self.N):
		# 			neuron = Vertex('X' + str(self.number.nextn()))
		# 			V[neuron.name] = neuron 
		# 			const = (Real('X' + str(self.number.nextn())), "Float")
		# 			newvar = ADD(newvar, MULT(const, (neuron, "Neuron")))

		# 		self.os.V[n[0]].symmap[n.elem.name] = newvar
		# 		self.os.C.append(oldvar == self.os.convertToZ3(newvar))

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

	def visitArgmaxOp(self, node: AST.ArgmaxOpNode):
		self.visit(node.expr)
		self.visit(node.func)

	def visitMaxOpList(self, node: AST.MaxOpListNode):
		self.visit(node.expr)
	
	def visitMaxOp(self, node: AST.MaxOpNode):
		self.visit(node.expr1)
		self.visit(node.expr2)

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


	def visitListOp(self, node: AST.ListOpNode):
		self.visit(node.expr)

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


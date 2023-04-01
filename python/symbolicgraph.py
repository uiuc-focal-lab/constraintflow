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

	def getMember(self, l, i):
		member = Real('m' + str(self.number.nextn()))
		conds = []
		for j in len(l):
			conds.append(Implies(i==j, member==l[j]))
		return member, conds

	def getLength(self, l, f):
		length = 0
		for i in range(len(l)):
			length += If(f(i, l[i]), 1, 0)
		return length 

	def visitDot(self, node):
		self.visit(node.left)
		self.visit(node.right)
		left = self.os.visit(node.left)
		right = self.os.visit(node.right)
		conds = []
		length = min(len(left.elist), len(right.elist))
		e1 = [None]*length
		e2 = [None]*length
		prev1 = -1
		prev2 = -1
		for i in range(length):
			e1[i] = Int('Index'+str(self.number.nextn()))
			e2[i] = Int('Index'+str(self.number.nextn()))
			cond1 = -1
			j = len(left.elist)
			while(j > 0):
				member, c = self.getMember(left.elist, prev1+j)
				cond1 = If(left.elist_func(prev1+j, member), prev1+j, cond1)
				conds += c
				j -= 1
			cond2 = -1
			j = len(right.elist)
			while(j > 0):
				member, c = self.getMember(right.elist, prev1+j)
				cond2 = If(right.elist_func(prev1+j, member), prev1+j, cond2)
				conds += c
				j -= 1
			conds += [cond1 == e1[i], cond2 == e2[i]]
			prev1 = e1[i]
			prev2 = e2[i]
		length1 = getLength(left.elist, left.elist_func)
		length2 = getLength(right.elist, right.elist_func)
		self.C.append(length1==length2)
		out = Real('out'+str(self.number.nextn()))
		sum = 0
		for i in range(length):
			conds.append(Implies(length1==i, out==sum))
			mem1, c1 = getMember(left.elist, e1[i])
			mem2, c2 = getMember(right.elist, e2[i])
			sum += mem1 * mem2
			conds += c1
			conds += c2 
		self.C += conds 
		self.M[DOT(left, right)] = out

	def visitGetMetadata(self, node):
		self.visit(node.expr)
		n = self.os.visit(node.expr)
		if(not isinstance(n, LIST)):
			if not node.metadata.name in self.V[n[0]].symmap.keys():
				newvar = None
				if(node.metadata.name == "bias"):
					newvar = (Real('X' + str(self.number.nextn())), "Bool")
				elif(node.metadata.name == "weight"):
					f = Function('f'+str(self.number.nextn()))
					newvar_list = [None]*(self.N)
					for i in range(self.N):
						newvar_list[i] = (Real('X' + str(self.number.nextn())), "Float")
						self.C.append(f(i, newvar[i]) == True)
					newvar = LIST(newvar_list, f)
				elif(node.metadata.name == "layer"):
					newvar = (Int('X' + str(self.number.nextn())), "Int")
				elif(node.metadata.name == "serial"):
					newvar = (Int('X' + str(self.number.nextn())), "Int")
				elif(node.metadata.name == "local_serial"):
					newvar = (Int('X' + str(self.number.nextn())), "Int")
				self.V[n[0]].symmap[node.metadata.name] = newvar
		else:
			f = Function('f'+ str(self.number.nextn()))
			final_list = []
			for i, ni in enumerate(n.elist):
				if not node.metadata.name in self.V[ni[0]].symmap.keys():
					newvar = None
					if(node.metadata.name == "bias"):
						newvar = (Real('X' + str(self.number.nextn())), "Bool")
					# elif(node.metadata.name == "weight"):
					# 	newvar = [(Real('X' + str(self.number.nextn())), "Float") for i in self.N]
					elif(node.metadata.name == "layer"):
						newvar = (Int('X' + str(self.number.nextn())), "Int")
					self.V[ni[0]].symmap[node.metadata.name] = newvar
				final_list.append(self.V[ni[0]].symmap[node.metadata.name])
				self.C.append(f(i, self.V[ni[0]].symmap[node.metadata.name]) == n.elist_func(i, ni))

			self.M[GETMETADAT(n, node.metadat.name)] = LIST(final_list, f)

	def visitGetElement(self, node):
		self.visit(node.expr)
		n = self.os.visit(node.expr)
		if(not isinstance(n, LIST)):
			if not node.elem.name in self.V[n[0]].symmap.keys():
				newvar = None
				if(self.shape[node.elem.name] == "Bool"):
					newvar = (Bool(str(node.elem.name) + '_X' + str(self.number.nextn())), "Bool")
				elif(self.shape[node.elem.name] == "Int"):
					newvar = (Int(str(node.elem.name) + '_X' + str(self.number.nextn())), "Int")
				else:
					newvar = (Real(str(node.elem.name) + '_X' + str(self.number.nextn())), "Float")
				self.V[n[0]].symmap[node.elem.name] = newvar
		else:
			f = Function('f'+ str(self.number.nextn()))
			final_list = []
			for i, ni in enumerate(n.elist):
				if not node.elem.name in self.V[ni[0]].symmap.keys():
					newvar = None
					
					if(self.shape[node.elem.name] == "Bool"):
						newvar = (Bool(str(node.elem.name) + '_X' + str(self.number.nextn())), "Bool")
					elif(self.shape[node.elem.name] == "Int"):
						newvar = (Int(str(node.elem.name) + '_X' + str(self.number.nextn())), "Int")
					else:
						newvar = (Real(str(node.elem.name) + '_X' + str(self.number.nextn())), "Float")
					self.V[ni[0]].symmap[node.elem.name] = newvar
				final_list.append(self.V[ni[0]].symmap[node.elem.name])
				self.C.append(f(i, self.V[ni[0]].symmap[node.elem.name]) == n.elist_func(i, ni))
			self.M[GETELEMENT(n, node.elem.name)] = LIST(final_list, f)

	def visitExprList(self, node):
		for e in node.exprlist:
			self.visit(e)

	def visitMap(self, node):
		self.visit(node.expr)
		checkPoly(self.os).visit(node.expr)
		p = self.os.visit(node.expr)
		p_poly = self.os.convertToPoly(p)
		for n in p_poly.coeffs.keys():
			if isinstance(node.func, AST.VarNode):
				elist = []
			else:	
				elist = self.os.visit(node.func)
			elist = AST.ExprListNode(elist + [n, p_poly.coeffs[n]])
			fcall = AST.FuncCallNode(node.func, elist)
			self.visitFuncCall(fcall, True)

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


	def initV(self, v1):
		if(not "bias" in v1.symmap.keys()):
			v1.symmap["bias"] = (Real('X' + str(self.number.nextn())), "Float")
		if(not "layer" in v1.symmap.keys()):
			v1.symmap["layer"] = (Int('X' + str(self.number.nextn())), "Int")
		if(not "weight" in v1.symmap.keys()):
			v1.symmap["weight"] = [(Real('X' + str(self.number.nextn())), "Float") for i in range(self.N)]
		if(not "local_serial" in v1.symmap.keys()):
			v1.symmap["local_serial"] = (Int('X' + str(self.number.nextn())), "Int")
		if(not "serial" in v1.symmap.keys()):
			v1.symmap["serial"] = (Int('X' + str(self.number.nextn())), "Int")

		for s in self.shape.keys():
			if(not s in v1.symmap.keys()):
				if(self.shape[s] == "Bool"):
					newvar = (Bool('X' + str(self.number.nextn())), "Bool")
				elif(self.shape[s] == "Int"):
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

	def visitArgmaxOp(self, node):
		self.visit(node.expr)
		e = self.os.visit(node.expr)

		# if(node.op == "max"):
		# 	newvar = Real('X' + str(self.number.nextn()))
		# 	self.os.M[MAX(e)] = (newvar, "Float")
		# 	t = False
		# 	for i in range(len(e)):
		# 		self.os.C.append(newvar >= self.os.convertToZ3(e[i]))
		# 		t = Or(t, newvar == self.os.convertToZ3(e[i]))
		# 	self.os.C.append(t)
		# elif(node.op == "min"):
		# 	newvar = Real('X' + str(self.number.nextn()))
		# 	self.os.M[MIN(e)] = (newvar, "Float")
		# 	t = False
		# 	for i in range(len(e)):
		# 		self.os.C.append(newvar <= self.os.convertToZ3(e[i]))
		# 		t = Or(t, newvar == self.os.convertToZ3(e[i]))
		# 	self.os.C.append(t)
		if isinstance(node.func, AST.VarNode):
			pre_elist = []
		else:	
			pre_elist = self.os.visit(node.func)
		for n in e.elist:
			elist = AST.ExprListNode(pre_elist + [n, n])
			fcall = AST.FuncCallNode(node.func, elist)
			self.visitFuncCall(fcall, True)
		new_list = copy(e.elist)
		f = Function('f'+str(self.number.nextn()), IntSort(), RealSort(), BoolSort())
		for i in range(e.elist):
			cond = True 
			for j in range(e.elist):
				if i!=j:
					if(node.op == "argmax"):
						elist = AST.ExprListNode(pre_elist + [e.elist[i], e.elist[j]])
					else:
						elist = AST.ExprListNode(pre_elist + [e.elist[j], e.elist[i]])
					fcall = AST.FuncCallNode(node.func, elist)
					r = self.os.visit(fcall)
					cond = AND(cond, r)
			self.C.append(f(i, e.elist[i]) == convertToZ3(cond) )
		if(node.op == "argmax"):
			self.os.M[ARGMAX(e, node.func.name)] = LIST(new_list, f)
		else:
			self.os.M[ARGMIN(e, node.func.name)] = LIST(new_list, f)

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
			self.os.C.append(If(self.os.convertToZ3(c), self.compareV(self.os.V[left[0]], newvar), self.compareV(self.os.V[right[0]], newvar)))
		else:
			newvar = Real('X' + str(self.number.nextn()))
			self.os.M[Ternary(c, left, right)] = (newvar, "Float")
			self.os.C.append(If(self.os.convertToZ3(c), self.os.convertToZ3(left) == newvar, self.os.convertToZ3(right) == newvar))

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

			newvar = ADD(newvar, Mult(const, (neuron.name, "Neuron")))
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

		self.os.M[Traverse(e, node.direction, p_name, s_name, node.func.name)] = (temp_var, "Float") 

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
				newvar = (Real('X' + str(self.number.nextn())), "Float")

				for i in range(self.N):
					neuron = Vertex('X' + str(self.number.nextn()))
					V[neuron.name] = neuron 
					const = (Real('X' + str(self.number.nextn())), "Float")
					newvar = ADD(newvar, Mult(const, (neuron, "Neuron")))

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


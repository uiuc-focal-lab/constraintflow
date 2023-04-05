import astVisitor
import ast as AST
from z3 import *
from value import *
from symbolicos import *
import copy

def populate_vars(vars, v, C, store, os, constraint, number, flag = True):
	print(v.name.decl().name())
	for var in vars.keys():
		if not var in v.symmap.keys():
			if(vars[var] == "Bool"):
				v.symmap[var] = (Bool(v.name.decl().name() + "_" + var + "_" + str(number.nextn())), vars[var])
			elif(vars[var] == "Int"):
				v.symmap[var] = (Int(v.name.decl().name() + "_" + var + "_" + str(number.nextn())), vars[var])
			elif(vars[var] == 'ZonoExp'):
				sum = (Real(v.name.decl().name() + "_const_" + str(number.nextn())), 'Float')
				for i in range(os.N):
					coeff = (Real(v.name.decl().name() + "_coeff_" + str(number.nextn())), 'Float')
					noise = Real(v.name.decl().name() + "_noise_" + str(number.nextn()))
					sum = ADD(sum, MULT(coeff, (noise, 'Noise')))
					C.append(noise <= 1)
					C.append(noise >= -1)
				v.symmap[var] = sum 
			else:
				v.symmap[var] = (Real(v.name.decl().name() + "_" + var + "_" + str(number.nextn())), vars[var])

	store["curr_new"] = (v.name, "Neuron")
	# print(store)
	# print(os.store)
	c = os.convertToZ3(os.visit(constraint))
	del store["curr_new"]
	if flag:
		C.append(c)
	return c

class SymbolicGraph(astVisitor.ASTVisitor):

	def __init__(self, store, F, constraint, shape, N, number, M, V, C):
		self.M = M
		self.V = V 
		self.C = C 
		self.constraint = constraint
		self.store = store
		self.F = F
		self.shape = shape
		self.N = N 
		self.os = SymbolicOperationalSemantics(self.store, self.F, self.M, self.V, self.C, self.shape, self.N)
		self.number = number 
		self.currop = None #Stores the relationship between curr and prev for traverse proof
		g = getVars(self.constraint, self.shape)
		g.visit(self.constraint)
		self.vars = g.vars

	def visitInt(self, node):
		pass

	def visitEpsilon(self, node):
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
			# print(val)
			# print(self.os.get_type(val))
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
		# e = self.os.visit(node.expr)



		# checkPoly(self.os, self.vars, self.constraint).visit(node.expr)
		# print(node.expr)
		# count = 0 
		# for v in self.V:
		# 	if count==6:
		# 		print(self.V[v].symmap)
		# 	count+=1
		p = self.os.visit(node.expr)
		self.get_map(p, node)

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

	def check_invariant(self, node):
		# newV = copy.copy(self.os.V)
		# newM = copy.copy(self.os.M)
		newC = copy.copy(self.os.C)
		# newStore = copy.copy(self.store)
		
		# oldV = copy.copy(self.os.V)
		# # oldM = copy.copy(self.os.M)
		# oldC = copy.copy(self.os.C)
		# oldStore = copy.copy(self.store)

		const = Real('X' + str(self.number.nextn()))
		input = (const, "Float")
		output = (const, "Float")
		for i in range(self.N):
			neuron = Vertex('V' + str(self.number.nextn()))
			self.os.V[neuron.name] = neuron 
			populate_vars(self.vars, neuron, newC, self.store, self.os, self.constraint, self.number)
			coeff = (Real('X' + str(self.number.nextn())), "Float")
			if isinstance(node.stop, AST.VarNode):
				elist_stop = []
			else:	
				elist_stop = self.os.visit(node.stop)
			if(isinstance(node.stop, AST.VarNode)):
				self.visitFuncCall(AST.FuncCallNode(node.stop, AST.ExprListNode(elist_stop + [(neuron.name, "Neuron"), coeff])), True)
			
			if isinstance(node.func, AST.VarNode):
				elist_func = []
			else:	
				elist_func = self.os.visit(node.func)
			if(isinstance(node.func, AST.VarNode)):
				self.visitFuncCall(AST.FuncCallNode(node.func, AST.ExprListNode(elist_func + [(neuron.name, "Neuron"), coeff])), True)

			if(isinstance(node.stop, AST.VarNode)):
				val_stop = self.os.visitFuncCall(AST.FuncCallNode(node.stop, AST.ExprListNode(elist_stop + [(neuron.name, "Neuron"), coeff])), True)
			else: #expression like True
				val_stop = self.os.visit(node.stop)

			if(isinstance(node.func, AST.VarNode)):
				val_func = self.os.visitFuncCall(AST.FuncCallNode(node.func, AST.ExprListNode(elist_func + [(neuron.name, "Neuron"), coeff])), True)
			else: #expression like True
				val_func = self.os.visit(node.func)

			input = ADD(input, MULT(coeff, (neuron.name, "Neuron")))
			output = self.os.get_binop(output, IF(val_stop, val_func, MULT(coeff, (neuron.name, 'Neuron'))), ADD)

		s = Solver()
		old_val = self.os.store[node.expr.name]
		self.os.store[node.expr.name] = input 
		p_input = self.os.convertToZ3(self.os.visit(node.p))

		self.os.store[node.expr.name] = output  
		p_output = self.os.convertToZ3(self.os.visit(node.p))
		
		# self.os.store[node.expr.name] = old_val

		p = Not(Implies(And(And(newC), p_input), p_output))
		s.add(p)
		if(not (s.check() == unsat)):
			raise Exception("Induction step is not true")
		else:
			print("Induction step proved")
		
		# self.os.M = oldM
		# self.os.V = oldV
		# self.os.C = oldC
		# self.os.store = oldStore

		# self.M = self.os.M
		# self.V = self.os.V
		# self.C = self.os.C 
		# self.store = self.os.store 

		const = Real('X' + str(self.number.nextn()))
		output_poly = (const, "Float")
		for i in range(self.N):
			neuron = Vertex('V' + str(self.number.nextn()))
			self.os.V[neuron.name] = neuron 
			populate_vars(self.vars, neuron, self.os.C, self.store, self.os, self.constraint, self.number)
			coeff = (Real('c' + str(self.number.nextn())), "Float")
			output_poly = ADD(output_poly, MULT(coeff, (neuron.name, "Neuron")))

		# print(type(output))
		#self.C.append(self.os.convertToZ3(output) == self.os.convertToZ3(output_poly))
		self.os.store[node.expr.name] = output_poly 
		p_output = self.os.convertToZ3(self.os.visit(node.p))
		self.os.store[node.expr.name] = old_val
		self.os.C.append(p_output)

		return output_poly, p_output

	def visitTraverse(self, node):
		self.visit(node.expr)
		e = self.os.visit(node.expr)
		p = self.os.visit(node.p)
		
		pz3 = self.os.convertToZ3(p)
		prop = Not(Implies(And(And(self.os.C), self.currop), pz3))
		s = Solver()
		s.add(prop)
		if(not (s.check() == unsat)):
			raise Exception("Invariant is not true on input")

		# old_val = self.os.store[node.expr.name]
		output, prop_output = self.check_invariant(node)
		self.C.append(prop_output)

		if(isinstance(node.priority, AST.VarNode)):
			p_name = node.priority.name
		else:
			p_name = self.os.visit(node.priority)

		if(isinstance(node.stop, AST.VarNode)):
			s_name = node.stop.name
		else:
			s_name = self.os.visit(node.stop)

		self.os.M[TRAVERSE(e, node.direction, p_name, s_name, node.func.name)] = output  

	def visitTransRetBasic(self, node):
		self.visit(node.exprlist)

	def visitTransRetIf(self, node):
		self.visit(node.cond)
		self.visit(node.tret)
		self.visit(node.fret)


class checkPoly(astVisitor.ASTVisitor):
	
	def __init__(self, os, vars, constraint):
		self.os = os
		self.vars = vars 
		self.constraint = constraint

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
						neuron = Vertex('V' + str(self.number.nextn()))
						V[neuron.name] = neuron
						populate_vars(self.vars, neuron, self.os.C, self.os.store, self.os, self.constraint, self.number)
						const = (Real('c' + str(self.number.nextn())), "Float")
						newvar = ADD(newvar, MULT(const, (neuron.name, "Neuron")))

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
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print(n)
		get_getElement(n, node)

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
		n = node.metadata.name 
		if n=='weight':
			type = 'Float list'
		elif n == 'bias':
			type = 'Float'
		else:
			type = 'Int'
		self.vars[n] = type 

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


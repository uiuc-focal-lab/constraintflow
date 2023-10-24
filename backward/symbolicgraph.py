import astVisitor
import astcf as AST
from z3 import *
#from cvc5.pythonic import * 
from value import *
from symbolicos import *
from optimization import *
import copy
import time
import astPrinter
from solver import Opt_solver


def populate_vars(vars, v, C, store, os, constraint, number, flag = True):
	for var in vars.keys():
		if not var in v.symmap.keys():
			#vname = str(v.name) #uncomment when using cvc5
			vname = v.name.decl().name() #uncomment when using Z3
			if(vars[var] == "Bool" or vars[var] == "Ct"):
				v.symmap[var] = (Bool(vname + "_" + var + "_" + str(number.nextn())), vars[var])
			elif(vars[var] == "Int"):
				v.symmap[var] = (Int(vname + "_" + var + "_" + str(number.nextn())), vars[var])
			else:
				v.symmap[var] = (Real(vname + "_" + var + "_" + str(number.nextn())), vars[var])
			'''
			elif(vars[var] == 'ZonoExp'):
				sum = (Real(vname + "_const_" + str(number.nextn())), 'Float')
				for i in range(os.Nzono):
					coeff = (Real(vname + "_coeff_" + str(number.nextn())), 'Float')
					if len(os.old_eps)==i:
						noise = Real("noise_" + str(number.nextn()))
						os.old_eps.append(noise)
						C.append(noise <= 1)
						C.append(noise >= -1)
					noise = os.old_eps[i]
					sum = ADD(sum, MULT(coeff, (noise, 'Noise')))
				v.symmap[var] = sum 
			'''


	store["curr_new"] = (v.name, "Neuron")
	Ctemp = []
	if(not isinstance(constraint, list)):
		constraint = [constraint]
	for cons in constraint:
		visitedc = os.visit(cons)
		if(not flag):
			os.flag = True

		Ctemp.append(os.convertToZ3(visitedc))
		if(not flag):
			os.flag = False
		
	del store["curr_new"]
	if flag:
		C += Ctemp
	return Ctemp

class SymbolicGraph(astVisitor.ASTVisitor):

	def __init__(self, store, F, constraint, shape, Nprev, Nzono, number, M, V, C, E, old_eps, old_neurons, solver):
		self.M = M
		self.V = V 
		self.C = C 
		self.E = E 
		self.old_eps = old_eps 
		self.old_neurons = old_neurons 
		self.constraint = constraint
		self.store = store
		self.F = F
		self.shape = shape
		self.Nprev = Nprev
		self.Nzono = Nzono
		self.os = SymbolicOperationalSemantics(self.store, self.F, self.M, self.V, self.C, self.E, self.old_eps, self.old_neurons, self.shape, self.Nprev, self.Nzono)
		self.number = number 
		self.currop = None #Stores the relationship between curr and prev for traverse proof
		g = getVars(self.constraint, self.shape)
		g.visit(self.constraint)
		self.vars = g.vars
		self.flag = False
		#set_param("timeout", 30)
		self.solver = solver 

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

	def visitConcat(self, node):
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
						newvar_list = [None]*(self.Nprev)
						for i in range(self.Nprev):
							newvar_list[i] = (Real('X' + str(self.number.nextn())), "Float")
						newvar = newvar_list
					elif(node.metadata.name == "layer"):
						newvar = (Int('X' + str(self.number.nextn())), "Int")
					elif(node.metadata.name == "serial"):
						newvar = (Int('X' + str(self.number.nextn())), "Int")
					elif(node.metadata.name == "local_serial"):
						newvar = (Int('X' + str(self.number.nextn())), "Int")
					elif(node.metadata.name == "equations"):
						newvar_list = [None]*(self.Nprev)
						for i in range(self.Nprev):
							newvar_list[i] = (Real('X' + str(self.number.nextn())), "PolyExp")
						newvar = newvar_list
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

	def visitGetElementAtIndex(self, node):
		self.visit(node.expr)
		# n = self.os.visit(node.expr)
		# self.get_getElement(n, node)

	def visitExprList(self, node):
		for e in node.exprlist:
			self.visit(e)
			
	def get_map(self, val, node):
		#print('jhsgdjfhakjshdgfkjhaskdhfg')
		#print(time.time())
		if isinstance(val, IF):
			self.get_map(val.left, node)
			self.get_map(val.right, node)
		else:
			#print(time.time())
			u_type = self.os.get_type(val)
			#print(uiuiui)
			if u_type=='PolyExp' or u_type=='Neuron':
				expr = self.os.convertToPoly(val)
			else:
				expr = self.os.convertToZono(val)
			# print('jhsgdjfhakjshdgfkjhaskdhfg')
			#print(time.time())

			for n in expr.coeffs.keys():
				if isinstance(node.func, AST.VarNode):
					elist = []
				else:	
					elist = self.os.visit(node.func)
				elist = AST.ExprListNode(elist + [n, expr.coeffs[n]])
				fcall = AST.FuncCallNode(node.func, elist)
				# if i==0: 
				# 	print('jhsdgjhfgjhsdgf')
				# 	print(time.time())
				self.visitFuncCall(fcall, True)
				# if i==0:
			#print(time.time())


				# i = i+1

	def visitMap(self, node):
		checkPoly(self.os, self.vars, self.constraint, self.number, self.Nzono).visit(node.expr)
		self.visit(node.expr)
		p = self.os.visit(node.expr)
		self.get_map(p, node)

	def get_maplist(self, val, node):
		if isinstance(val, IF):
			self.get_maplist(val.left, node)
			self.get_maplist(val.right, node)
		else:
			for n in val:
				if isinstance(node.func, AST.VarNode):
					elist = []
				else:	
					elist = self.os.visit(node.func)
				elist = AST.ExprListNode(elist + [n])
				fcall = AST.FuncCallNode(node.func, elist)
				self.visitFuncCall(fcall, True)

	def visitMapList(self, node):
		#checkPoly(self.os, self.vars, self.constraint, self.number, self.Nzono).visit(node.expr)
		self.visit(node.expr)
		p = self.os.visit(node.expr)
		self.get_maplist(p, node)

	def visitFuncCall(self, node, preeval = False):
		name = node.name.name
		if(isinstance(name, str)): 
			func = self.F[node.name.name]
		else:
			func = self.F[node.name.name.name]

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

	def visitLp(self, node):
		self.visit(node.expr)
		self.visit(node.constraints)
		expr = self.os.visit(node.expr)
		constraints = self.os.convertToZ3(self.os.visit(node.constraints))
		out = Real('Lp_'+str(self.number.nextn()))
		self.M[LP(node.op, expr, constraints)] = out
		if(node.op == "maximize"):
			self.C.append(Implies(self.os.convertToZ3(constraints), out >= self.os.convertToZ3(expr)))
		else:
			self.C.append(Implies(self.os.convertToZ3(constraints), out <= self.os.convertToZ3(expr)))

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
		# newC = copy.copy(self.os.C)
		# newStore = copy.copy(self.store)
		
		# oldV = copy.copy(self.os.V)
		# # oldM = copy.copy(self.os.M)
		# oldC = copy.copy(self.os.C)
		# oldStore = copy.copy(self.store)

		const = Real('inv_X' + str(self.number.nextn()))
		input = (const, "Float")
		output = (const, "Float")
		for i in range(self.Nzono):
			if len(self.old_neurons)==i:
				neuron = Vertex('V' + str(self.number.nextn()))
				self.os.V[neuron.name] = neuron 
				populate_vars(self.vars, neuron, self.os.C, self.store, self.os, self.constraint, self.number)
				self.old_neurons.append(neuron)
			neuron = self.old_neurons[i]
			coeff = (Real('inv_X' + str(self.number.nextn())), "Float")
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
		# s = Solver()
		old_val = self.os.store[node.expr.name]
		self.os.store[node.expr.name] = input 
		p_input = self.os.convertToZ3(self.os.visit(node.p))

		self.os.store[node.expr.name] = output  
		p_output = self.os.convertToZ3(self.os.visit(node.p))
		# print((p_output))
		# dfhg
		# self.os.store[node.expr.name] = old_val
		lhs = And(self.os.C + [p_input]+ self.os.tempC)
		rhs = p_output
		# print(lhs)
		# print(rhs.children()[1])
		# solver = Opt_solver()
		w = self.solver.solve(lhs, rhs)
		print("end",time.time())
		if w:
			print('Induction step proved')
		else:
			raise Exception("Induction step is not true")
		# p = Not(Implies(lhs, p_output))
		# flag = True 
		# opt = None
		# if opt:
		# 	d = p_output.decl()
		# 	flag = True 
		# 	for i in range(len(opt)):
		# 		solver = Solver()
		# 		p = Not(Implies(And(lhs), d(opt[i][0], opt[i][1])))
		# 		print(p)
		# 		solver.add(p)
		# 		if(not (solver.check() == unsat)):
		# 			flag = False 
		# 			dhgdhgfd
		# 			break 
		# 		else:
		# 			print('proved ', i)
		# else:
		# 	flag = False 
		# print(flag)

		# s.add(p)
		# print("gen",time.time())
		# #set_param('timeout', 30)
		# #print("timeout place")
		# #s.set("timeout",10000)
		# #print(p)
		# #z3.set_option(timeout=30)
		# print(p)
		# jhdsg
		# if(not (s.check() == unsat)):
		# 	print(p)
		# 	print(s.model())
		# 	print("end",time.time())
		# 	raise Exception("Induction step is not true")
		# else:
		# 	print("Induction step proved")
		
		# print("end",time.time())
		
		# self.os.M = oldM
		# self.os.V = oldV
		# self.os.C = oldC
		# self.os.store = oldStore

		# self.M = self.os.M
		# self.V = self.os.V
		# self.C = self.os.C 
		# self.store = self.os.store 

		const = Real('out_trav_X' + str(self.number.nextn()))
		output_poly = (const, "Float")
		for i in range(self.Nzono):
			if len(self.os.old_neurons)==i:
				neuron = Vertex('V' + str(self.number.nextn()))
				self.os.V[neuron.name] = neuron 
				populate_vars(self.vars, neuron, self.os.C, self.store, self.os, self.constraint, self.number)
				self.os.old_neurons.append(neuron)
			neuron = self.os.old_neurons[i]
			coeff = (Real('out_trav_c' + str(self.number.nextn())), "Float")
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
		prop = Not(Implies(And(self.os.C + [self.currop]+self.os.tempC), pz3))
		s = Solver()
		#s.set("timeout",3000)
		s.add(prop)
		print("gen",time.time())
		# lhs = And(self.os.C + [self.currop]+self.os.tempC)
		# rhs = pz3 

		#print("induction query:")
		#print(prop)
		
		if(not (s.check() == unsat)):
			print("end",time.time())
			raise Exception("Invariant is not true on input")
		else:
			print("Invariant true on input")
		
		print("end",time.time())

		output, prop_output = self.check_invariant(node)

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
	
	def __init__(self, os, vars, constraint, number, Nzono):
		self.os = os
		self.vars = vars 
		self.constraint = constraint
		self.number = number
		self.Nzono = Nzono

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
	
	def visitConcat(self, node):
		self.visit(node.left)
		self.visit(node.right)
	
	def visitFuncCall(self, node: AST.FuncCallNode):
		name = node.name.name
		self.visit(F[name].expr)
		self.visit(node.arglist)

	def get_getMetadata(self, val, node):
		if isinstance(val, list):
			for v in val:
				self.get_getMetadata(v, node)
		elif isinstance(val, IF):
			self.get_getgetMetadata(val.left, node)
			self.get_getgetMetadata(val.right, node)
		else:
			name = node.metadata.name
			if name == "equations":
				newlist = []
				for eq in (self.os.V[val[0]].symmap[name]):
					if(isinstance(eq, tuple)):

						oldvar = eq[0]
						newvar = (Real('X' + str(self.number.nextn())), "Float")

						for i in range(self.Nzono):
							if len(self.os.old_neurons) == i:
								neuron = Vertex('V' + str(self.number.nextn()))
								self.os.old_neurons.append(neuron)
								self.os.V[neuron.name] = neuron
								populate_vars(self.vars, neuron, self.os.C, self.os.store, self.os, self.constraint, self.number)
							neuron = self.os.old_neurons[i]
							const = (Real('c' + str(self.number.nextn())), "Float")
							newvar = ADD(newvar, MULT(const, (neuron.name, "Neuron")))

						newlist.append(newvar)
						self.os.C.append(oldvar == self.os.convertToZ3(newvar))
				if newlist != []:
					self.os.V[val[0]].symmap[name] = newlist

	def visitGetMetadata(self, node: AST.GetMetadataNode):
		n = self.os.visit(node.expr)
		self.get_getMetadata(n, node)

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

					for i in range(self.Nzono):
						if len(self.os.old_neurons == i):
							neuron = Vertex('V' + str(self.number.nextn()))
							self.os.old_neurons.append(neuron)
							self.os.V[neuron.name] = neuron
							populate_vars(self.vars, neuron, self.os.C, self.os.store, self.os, self.constraint, self.number)
						neuron = self.os.old_neurons[i]
						const = (Real('c' + str(self.number.nextn())), "Float")
						newvar = ADD(newvar, MULT(const, (neuron.name, "Neuron")))

					self.os.V[val[0]].symmap[name] = newvar
					self.os.C.append(oldvar == self.os.convertToZ3(newvar))
				elif (self.os.V[val[0]].symmap[name])[1] == 'ZonoExp':
					oldvar = self.os.V[val[0]].symmap[name][0]
					newvar = (Real('X' + str(self.number.nextn())), "Float")

					for i in range(self.Nzono):
						# epsilon = Real('eps' + str(self.number.nextn()))
						if len(self.os.old_eps == i):
							epsilon = Real('eps_'+str(self.number.nextn()))
							self.os.old_eps.append(epsilon)
							self.os.C.append(epsilon <= 1)
							self.os.C.append(epsilon >= -1)
						epsilon = self.os.old_eps[i]
						const = (Real('X' + str(self.number.nextn())), "Float")
						newvar = ADD(newvar, MULT(const, (epsilon, "Noise")))
						

					self.os.V[val[0]].symmap[name] = newvar
					self.os.C.append(oldvar == self.os.convertToZ3(newvar))
			return

	def visitGetElement(self, node):
		n = self.os.visit(node.expr)
		# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		# print(n)
		self.get_getElement(n, node)

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

	def visitConcat(self, node):
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

	


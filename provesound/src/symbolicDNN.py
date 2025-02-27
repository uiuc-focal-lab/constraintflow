from z3 import *

from constraintflow import astVisitor
from constraintflow import astcf as AST

from provesound.src.value import *
from provesound.src.symbolicSemantics import *
from provesound.lib.globals import *


# The following class implements the symbolic DNN expansion and uses the symbolic 
# semantics of ConstraintFlow. Given a symbolic DNN, it produces the expanded symbolic DNN.
# The visitor pattern implements the inference rules described in Section 5.2 of the paper
# and Appendix G.1.
class SymbolicDNN(astVisitor.ASTVisitor):
	def __init__(self, store, F, constraint, shape, Nprev, Nsym, number, M, V, C, E, old_eps, old_neurons, solver, arrayLens, prevLength):
		# Initializes the SymbolicDNN visitor. The visitor takes in the function
		# store F, the soundness property (constraint), the shape, bound parameters 
		# Nprev and Nsym, the list of symbolic metadata (V), the list of new symbolic
		# variables (E), the list of existing symbolic variables (old_eps), the list of
		# existing symbolic neurons (old_neurons), the constraint solver, the dictionary of 
		# array lengths (arrayLens), and the maximum length of the previous layer. 
		
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
		self.Nsym = Nsym
		self.ss = SymbolicSemantics(self.store, self.F, self.M, self.V, self.C, self.E, self.old_eps, self.old_neurons, self.shape, self.Nprev, self.Nsym, arrayLens)
		self.number = number 
		self.currop = None #Stores the relationship between curr and prev for traverse proof
		g = getVars(self.constraint, self.shape)
		g.visit(self.constraint)
		self.vars = g.vars
		self.flag = False
		#set_param("timeout", 30)
		self.solver = solver 
		self.prevLength = prevLength

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
						self.ss.arrayLens[str(newvar_list)] = self.prevLength
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
						self.ss.arrayLens[str(newvar_list)] = self.prevLength
					self.V[n[0]].symmap[node.metadata.name] = newvar
			else:
				for ni in n:
					self.get_getMetadata(ni, node)

	def visitGetMetadata(self, node):
		self.visit(node.expr)
		n = self.ss.visit(node.expr)
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
		n = self.ss.visit(node.expr)
		self.get_getElement(n, node)

	def visitGetElementAtIndex(self, node):
		self.visit(node.expr)

	def visitExprList(self, node):
		for e in node.exprlist:
			self.visit(e)
			
	def get_map(self, val, node):
		if isinstance(val, IF):
			self.get_map(val.left, node)
			self.get_map(val.right, node)
		else:
			if isinstance(val, ADD) or isinstance(val, SUB):
				self.get_map(val.left, node)
				self.get_map(val.right, node)
			elif isinstance(val, MULT):
				lhstype = self.ss.get_type(val.left)
				rhstype = self.ss.get_type(val.right)

				if isinstance(node.func, AST.VarNode):
					elist = []
					fname = node.func
				else:	
					elist = self.visit(node.func)
					fname = node.func

				if(lhstype == 'PolyExp' or lhstype == 'SymExp' or lhstype == 'Neuron' or lhstype=='Noise'):
					elist = AST.ExprListNode(elist + [val.left, val.right])
					fcall = AST.FuncCallNode(fname, elist)
					self.visitFuncCall(fcall, True)
				elif(rhstype == 'PolyExp' or rhstype == 'SymExp' or rhstype == 'Neuron' or rhstype=='Noise'):
					elist = AST.ExprListNode(elist + [val.right, val.left])
					fcall = AST.FuncCallNode(fname, elist)
					self.visitFuncCall(fcall, True)
			elif isinstance(val, DIV):
				lhstype = self.ss.get_type(val.left)
				if isinstance(node.func, AST.VarNode):
					elist = []
					fname = node.func
				else:	
					elist = self.visit(node.func)
					fname = node.func

				if(lhstype == 'PolyExp' or lhstype == 'SymExp' or lhstype == 'Neuron' or lhstype=='Noise'):
					elist = AST.ExprListNode(elist + [val.left, DIV(1,val.right)])
					fcall = AST.FuncCallNode(fname, elist)
					self.visitFuncCall(fcall, True)
			else: 
				if(val[1] == 'Neuron' or val[1] == 'Noise'):
					if isinstance(node.func, AST.VarNode):
						elist = []
						fname = node.func
					else:	
						elist = self.visit(node.func)
						fname = node.func

					elist = AST.ExprListNode(elist + [val, 1])
					fcall = AST.FuncCallNode(fname, elist)
					self.visitFuncCall(fcall, True)

	def visitMap(self, node):
		self.visit(node.expr)
		expandSymbolicDNN(self.ss, self.vars, self.constraint, self.number, self.Nsym).visit(node.expr)
		p = self.ss.visit(node.expr)
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
					elist = self.ss.visit(node.func)
				elist = AST.ExprListNode(elist + [n])
				fcall = AST.FuncCallNode(node.func, elist)
				self.visitFuncCall(fcall, True)

	def visitMapList(self, node):
		self.visit(node.expr)
		p = self.ss.visit(node.expr)
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
			elist = self.ss.visit(node.arglist)
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
		const = Real('inv_X' + str(self.number.nextn()))
		input = (const, "Float")
		output = (const, "Float")
		for i in range(self.Nsym):
			if len(self.old_neurons)==i:
				neuron = Vertex('V' + str(self.number.nextn()))
				self.ss.V[neuron.name] = neuron 
				populate_vars(self.vars, neuron, self.ss.C, self.store, self.ss, self.constraint, self.number)
				self.old_neurons.append(neuron)
			neuron = self.old_neurons[i]
			coeff = (Real('inv_X' + str(self.number.nextn())), "Float")
			if isinstance(node.stop, AST.VarNode):
				elist_stop = []
			else:	
				elist_stop = self.ss.visit(node.stop)
			if(isinstance(node.stop, AST.VarNode)):
				self.visitFuncCall(AST.FuncCallNode(node.stop, AST.ExprListNode(elist_stop + [(neuron.name, "Neuron"), coeff])), True)
			elif(isinstance(node.stop, AST.FuncCallNode)):
				self.visitFuncCall(AST.FuncCallNode(node.stop.name, AST.ExprListNode(elist_stop + [(neuron.name, "Neuron"), coeff])), True)
			
			if isinstance(node.func, AST.VarNode):
				elist_func = []
			else:	
				elist_func = self.ss.visit(node.func)
			if(isinstance(node.func, AST.VarNode)):
				self.visitFuncCall(AST.FuncCallNode(node.func, AST.ExprListNode(elist_func + [(neuron.name, "Neuron"), coeff])), True)

			if(isinstance(node.stop, AST.VarNode)):
				val_stop = self.ss.visitFuncCall(AST.FuncCallNode(node.stop, AST.ExprListNode(elist_stop + [(neuron.name, "Neuron"), coeff])), True)
			elif(isinstance(node.stop, AST.FuncCallNode)):
				val_stop = self.ss.visitFuncCall(AST.FuncCallNode(node.stop.name, AST.ExprListNode(elist_stop + [(neuron.name, "Neuron"), coeff])), True)
			else: #expression like True
				val_stop = self.ss.visit(node.stop)

			if(isinstance(node.func, AST.VarNode)):
				val_func = self.ss.visitFuncCall(AST.FuncCallNode(node.func, AST.ExprListNode(elist_func + [(neuron.name, "Neuron"), coeff])), True)
			else: #expression like True
				val_func = self.ss.visit(node.func)

			input = ADD(input, MULT(coeff, (neuron.name, "Neuron")))
			output_temp = IF(val_stop, val_func, MULT(coeff, (neuron.name, 'Neuron')))
			cond = EQQ(coeff, (0, "Float"))
			output_temp = IF(cond, (0, 'Float'), output_temp)
			output = self.ss.get_binop(output, output_temp, ADD)

		old_val = self.ss.store[node.expr.name]
		self.ss.store[node.expr.name] = input 
		p_input = self.ss.convertToZ3(self.ss.visit(node.p))

		self.ss.store[node.expr.name] = output  
		p_output = self.ss.convertToZ3(self.ss.visit(node.p))
		lhs = And(self.ss.C + [p_input]+ self.ss.tempC)
		rhs = p_output
		update_generation_time()
		w = self.solver.solve(lhs, rhs)
		update_verification_time()
		# print("end",time.time())
		if w:
			pass
			# print('Induction step proved')
		else:
			raise Exception("Induction step is not true")
		

		const = Real('out_trav_X' + str(self.number.nextn()))
		output_poly = (const, "Float")
		for i in range(self.Nsym):
			if len(self.ss.old_neurons)==i:
				neuron = Vertex('V' + str(self.number.nextn()))
				self.ss.V[neuron.name] = neuron 
				populate_vars(self.vars, neuron, self.ss.C, self.store, self.ss, self.constraint, self.number)
				self.ss.old_neurons.append(neuron)
			neuron = self.ss.old_neurons[i]
			coeff = (Real('out_trav_c' + str(self.number.nextn())), "Float")
			output_poly = ADD(output_poly, MULT(coeff, (neuron.name, "Neuron")))

		self.ss.store[node.expr.name] = output_poly 
		p_output = self.ss.convertToZ3(self.ss.visit(node.p))
		self.ss.store[node.expr.name] = old_val
		self.ss.C.append(p_output)

		return output_poly, p_output

	def visitTraverse(self, node):
		self.visit(node.expr)
		e = self.ss.visit(node.expr)
		p = self.ss.visit(node.p)
		
		pz3 = self.ss.convertToZ3(p)
		prop = Not(Implies(And(self.ss.C + [self.currop]+self.ss.tempC), pz3))
		s = Solver()
		s.add(prop)
		update_generation_time()
		c = s.check()
		update_verification_time()
		# print("gen",time.time())
		
		if(not (c == unsat)):
			# print("end",time.time())
			raise Exception("Invariant is not true on input")
		# else:
			# print("Invariant true on input")
		
		# print("end",time.time())

		output, prop_output = self.check_invariant(node)

		if(isinstance(node.priority, AST.VarNode)):
			p_name = node.priority.name
		else:
			p_name = self.ss.visit(node.priority)

		if(isinstance(node.stop, AST.VarNode)):
			s_name = node.stop.name
		else:
			s_name = self.ss.visit(node.stop)

		self.ss.M[TRAVERSE(e, node.direction, p_name, s_name, node.func.name)] = output 

	def visitTransRetBasic(self, node):
		self.visit(node.exprlist)

	def visitTransRetIf(self, node):
		self.visit(node.cond)
		self.visit(node.tret)
		self.visit(node.fret)


class expandSymbolicDNN(astVisitor.ASTVisitor):
	
	def __init__(self, os, vars, constraint, number, Nsym):
		self.ss = os
		self.vars = vars 
		self.constraint = constraint
		self.number = number
		self.Nsym = Nsym

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
		self.visit(node.expr1)
		self.visit(node.expr2)

	def visitMaxOpList(self, node: AST.MaxOpListNode):
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
 
	def get_map(self, val, node):
		if isinstance(val, IF):
			self.get_map(val.left, node)
			self.get_map(val.right, node)
		else:
			if isinstance(val, ADD) or isinstance(val, SUB):
				self.get_map(val.left, node)
				self.get_map(val.right, node)
			elif isinstance(val, MULT):
				lhstype = self.ss.get_type(val.left)
				rhstype = self.ss.get_type(val.right)

				if isinstance(node.func, AST.VarNode):
					elist = []
					fname = node.func
				else:	
					elist = self.visit(node.func)
					fname = node.func

				if(lhstype == 'PolyExp' or lhstype == 'SymExp' or lhstype == 'Neuron' or lhstype=='Noise'):
					elist = AST.ExprListNode(elist + [val.left, val.right])
					fcall = AST.FuncCallNode(fname, elist)
					self.visitFuncCall(fcall, True)
				elif(rhstype == 'PolyExp' or rhstype == 'SymExp' or rhstype == 'Neuron' or rhstype=='Noise'):
					elist = AST.ExprListNode(elist + [val.right, val.left])
					fcall = AST.FuncCallNode(fname, elist)
					self.visitFuncCall(fcall, True)
			elif isinstance(val, DIV):
				lhstype = self.ss.get_type(val.left)
				if isinstance(node.func, AST.VarNode):
					elist = []
					fname = node.func
				else:	
					elist = self.visit(node.func)
					fname = node.func

				if(lhstype == 'PolyExp' or lhstype == 'SymExp' or lhstype == 'Neuron' or lhstype=='Noise'):
					elist = AST.ExprListNode(elist + [val.left, DIV(1,val.right)])
					fcall = AST.FuncCallNode(fname, elist)
					self.visitFuncCall(fcall, True)
			else: 
				if(val[1] == 'Neuron' or val[1] == 'Noise'):
					if isinstance(node.func, AST.VarNode):
						elist = []
						fname = node.func
					else:	
						elist = self.visit(node.func)
						fname = node.func

					elist = AST.ExprListNode(elist + [val, 1])
					fcall = AST.FuncCallNode(fname, elist)
					self.visitFuncCall(fcall, True)

	def visitMap(self, node):
		self.visit(node.expr)
		val = self.ss.visit(node.expr)
		self.get_map(val, node)

	def visitDot(self, node: AST.DotNode):
		self.visit(node.left)
		self.visit(node.right)
	
	def visitConcat(self, node):
		self.visit(node.left)
		self.visit(node.right)
	
	def visitFuncCall(self, node: AST.FuncCallNode, preeval=False):
		name = node.name.name
		if(isinstance(name, str)): 
			func = self.ss.F[node.name.name]
		else:
			func = self.ss.F[node.name.name.name]

		newvars = []
		oldvalues = {}

		if(not preeval):
			self.visit(node.arglist)
			elist = self.ss.visit(node.arglist)
		else:
			elist = node.arglist.exprlist

		for (exp,(t, arg)) in zip(elist, func.decl.arglist.arglist):
			if arg.name in self.ss.store.keys():
				oldvalues[arg.name] = self.ss.store[arg.name]
			else:
				newvars.append(arg.name)

			self.ss.store[arg.name] = exp

		self.visit(func.expr)
		
		for v in newvars:
			del self.ss.store[v]

		for ov in oldvalues.keys():
			self.ss.store[ov] = oldvalues[ov]


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
				for eq in (self.ss.V[val[0]].symmap[name]):
					if(isinstance(eq, tuple)):

						oldvar = eq[0]
						newvar = (Real('X' + str(self.number.nextn())), "Float")

						for i in range(self.Nsym):
							if len(self.ss.old_neurons) == i:
								neuron = Vertex('V' + str(self.number.nextn()))
								self.ss.old_neurons.append(neuron)
								self.ss.V[neuron.name] = neuron
								populate_vars(self.vars, neuron, self.ss.C, self.ss.store, self.ss, self.constraint, self.number)
							neuron = self.ss.old_neurons[i]
							const = (Real('c' + str(self.number.nextn())), "Float")
							newvar = ADD(newvar, MULT(const, (neuron.name, "Neuron")))

						newlist.append(newvar)
						self.ss.C.append(oldvar == self.ss.convertToZ3(newvar))
				if newlist != []:
					self.ss.V[val[0]].symmap[name] = newlist

	def visitGetMetadata(self, node: AST.GetMetadataNode):
		n = self.ss.visit(node.expr)
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
			if isinstance(self.ss.V[val[0]].symmap[name], tuple):
				if (self.ss.V[val[0]].symmap[name])[1] == 'PolyExp':
					oldvar = self.ss.V[val[0]].symmap[name][0]
					newvar = (Real('X' + str(self.number.nextn())), "Float")

					for i in range(self.Nsym):
						if len(self.ss.old_neurons) == i:
							neuron = Vertex('V' + str(self.number.nextn()))
							self.ss.old_neurons.append(neuron)
							self.ss.V[neuron.name] = neuron
							populate_vars(self.vars, neuron, self.ss.C, self.ss.store, self.ss, self.constraint, self.number)
						neuron = self.ss.old_neurons[i]
						const = (Real('c' + str(self.number.nextn())), "Float")
						newvar = ADD(newvar, MULT(const, (neuron.name, "Neuron")))

					self.ss.V[val[0]].symmap[name] = newvar
					self.ss.C.append(oldvar == self.ss.convertToZ3(newvar))
				elif (self.ss.V[val[0]].symmap[name])[1] == 'SymExp':
					oldvar = self.ss.V[val[0]].symmap[name][0]
					newvar = (Real('X' + str(self.number.nextn())), "Float")

					for i in range(self.Nsym):
						if len(self.ss.old_eps) == i:
							epsilon = Real('eps_'+str(self.number.nextn()))
							self.ss.old_eps.append(epsilon)
							self.ss.C.append(epsilon <= 1)
							self.ss.C.append(epsilon >= -1)
						epsilon = self.ss.old_eps[i]
						const = (Real('X' + str(self.number.nextn())), "Float")
						newvar = ADD(newvar, MULT(const, (epsilon, "Noise")))
						

					self.ss.V[val[0]].symmap[name] = newvar
					self.ss.C.append(oldvar == self.ss.convertToZ3(newvar))
			return

	def visitGetElement(self, node):
		n = self.ss.visit(node.expr)
		self.get_getElement(n, node)



# This class implements a visitor pattern to get the variables in a ConstraintFlow expression.
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
		self.visit(self.F[name].expr)


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

	

def populate_vars(vars, v, C, store, ss, constraint, number, flag = True):
	"""
	Populates the symbolic map of a vertex with the given variables.
	
	Parameters:
		vars (dict): A dictionary of variable names to their types.
		v (Vertex): The new vertex to populate.
		C (list): The list of constraints to add to.
		store (dict): The store of vertices.
		ss (SymbolicSemantics): An instance of the symbolic semantics.
		constraint (list or Constraint): The constraints to add to the vertex.
		number (Number): A counter for generating variable names.
		flag (bool): Whether to add the constraints to the global list of constraints.
		
	Returns:
		list: The list of z3 constraints added to the vertex.
	"""
	
	for var in vars.keys():
		if not var in v.symmap.keys():
			vname = v.name.decl().name()
			if(vars[var] == "Bool" or vars[var] == "Ct"):
				v.symmap[var] = (Bool(vname + "_" + var + "_" + str(number.nextn())), vars[var])
			elif(vars[var] == "Int"):
				v.symmap[var] = (Int(vname + "_" + var + "_" + str(number.nextn())), vars[var])
			else:
				v.symmap[var] = (Real(vname + "_" + var + "_" + str(number.nextn())), vars[var])
			
	store["curr_new"] = (v.name, "Neuron")
	Ctemp = []
	if(not isinstance(constraint, list)):
		constraint = [constraint]
	for cons in constraint:
		visitedc = ss.visit(cons)
		if(not flag):
			ss.flag = True
		Ctemp.append(ss.convertToZ3(visitedc))
		if(not flag):
			ss.flag = False
		
	del store["curr_new"]
	if flag:
		C += Ctemp
	return Ctemp
import astVisitor
import astcf as AST
from z3 import *
#from cvc5.pythonic import * 
from value import *
from symbolicos import *
from symbolicgraph import *
from solver import Opt_solver
import time
set_param('parallel.enable', True) #uncomment when using Z3

exptemp = None

# class AstRefKey:
#     def __init__(self, n):
#         self.n = n
#     def __hash__(self):
#         return self.n.hash()
#     def __eq__(self, other):
#         return self.n.eq(other.n)
#     def __repr__(self):
#         return str(self.n)

# def askey(n):
#     assert isinstance(n, AstRef)
#     return AstRefKey(n)

# def get_vars(f):
#     r = set()
#     def collect(f):
#       if is_const(f): 
#           if f.decl().kind() == Z3_OP_UNINTERPRETED and not askey(f) in r:
#               r.add(askey(f))
#       else:
#           for c in f.children():
#               collect(c)
#     collect(f)
#     return r

# x = Real('x')
# y = Real('y')
# plus = (x + y).decl()
# lt = (x<y).decl()
# le = (x<=y).decl()
# gt = (x>y).decl()
# ge = (x>=y).decl()
# eq = (x==y).decl()
# comparison = [lt, le, gt, ge, eq]

class Verify(astVisitor.ASTVisitor):

	def __init__(self):
		self.shape = {}
		self.F = {}
		self.theta = {}
		self.Nprev = 3
		self.Nzono = 3
		self.Ncurr = 2
		self.number = Number()
		self.M = {}
		self.V = {}
		self.C = []
		self.E = []
		self.old_eps = []
		self.old_neurons = []
		# for i in range(self.Nzono):
		# 	e = Real('eps_'+str(self.number.nextn()))
		# 	self.old_eps.append(e)
		# 	self.C.append(e <= 1)
		# 	self.C.append(e >= -1)
			# n = Real('V_'+str(self.number.nextn()))
			# self.old_neurons.append(n)
		self.store = {}
		#set_param("timeout", 30)

	def visitShapeDecl(self, node):
		for (t,e) in node.elements.arglist:
			self.shape[e.name] = t.name

		self.constraint = node.p

	def visitFunc(self, node):
		self.F[node.decl.name.name] = node

	def visitTransformer(self, node):
		self.theta[node.name.name] = node

	def visitFlow(self, node):
		global exptemp
		print()
		print("start !!!!!!!!!!!!!",time.time())
		node = self.theta[node.trans.name]
		for op_i in range(len(node.oplist.olist)):
			self.E.clear()

			hasE = node.oplist.opsE[op_i]
			op = node.oplist.olist[op_i]
			store = self.store
			if(op.op.op_name == "Relu"):
				nprev= 1
			else:
				nprev = self.Nprev
			s = SymbolicGraph(self.store, self.F, self.constraint, self.shape, nprev, self.Nzono, self.number, self.M, self.V, self.C, self.E, self.old_eps, self.old_neurons)
			s.os.hasE = hasE
			#node = self.theta[node.trans.name]
			if("curr" in node.arglist):
				curr = Vertex('Curr')
				self.V[curr.name] =  curr 
				populate_vars(s.vars, curr, self.C, self.store, s.os, self.constraint, self.number)
				store["curr"] = (curr.name, "Neuron")
			if("prev" in node.arglist):
				prev = []
				for i in range(nprev):
					p = Vertex('Prev'+str(i))
					prev.append((p.name, "Neuron"))
					self.V[p.name] = p
					populate_vars(s.vars, p, self.C, self.store, s.os, self.constraint, self.number)
				store["prev"] = prev 
			if("curr_list" in node.arglist):
				curr_list = []
				for i in range(self.Ncurr):
					p = Vertex('curr_list'+str(i))
					curr_list.append((p.name, "Neuron"))
					self.V[p.name] = p
					populate_vars(s.vars, p, self.C, self.store, s.os, self.constraint, self.number)
				store["curr_list"] = curr_list

		#for op_i in range(len(node.oplist.olist)):
			#self.E.clear()
			#op = node.oplist.olist[op_i]
			curr_prime = Vertex('curr_prime' + str(op_i))
			s.V[curr_prime.name] = curr_prime

			#Define relationship between curr and prev
			if(op.op.op_name == "Affine"):
				if(not "weight" in curr.symmap.keys()):
					curr.symmap["weight"] = [(Real('weight_curr' + str(op_i) + "_" + str(self.number.nextn())), "Float") for i in range(nprev)]
				if(not "bias" in curr.symmap.keys()):
					curr.symmap["bias"] = ((Real('bias_curr' + str(self.number.nextn())), "Float"))
				exptemp = (0, 'Float')
				for i in range(len(prev)):
					exptemp = ADD(exptemp, MULT(prev[i], curr.symmap["weight"][i]))
				exptemp = ADD(exptemp, curr.symmap["bias"])

				exptemp = s.os.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)
			elif(op.op.op_name == "rev_Affine"):
				if(not "equations" in curr.symmap.keys()):
					curr.symmap["equations"] = [(Real('equations_' + str(op_i) + "_" + str(self.number.nextn())), "PolyExp") for i in range(nprev)]
				exptemp = (True, "Bool")
				for t in curr.symmap["equations"]:
					exptemp = AND(exptemp, EQQ(curr.name, t))

				s.currop = s.os.convertToZ3(exptemp)
			elif(op.op.op_name == "Relu"):
				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = ADD(exptemp, prev[i])
				exptemp = IF(GEQ(exptemp, (0, 'Float')), exptemp, (0, 'Float'))

				exptemp = s.os.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)

			elif(op.op.op_name == "rev_Relu"):

				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = ADD(exptemp, prev[i])
				
				exptemp = s.os.convertToZ3(exptemp)
				prevexp = IF(GEQ(curr.name, (0, 'Float')), curr.name, (0, 'Float'))
				s.currop = ( s.os.convertToZ3(prevexp) == exptemp)

			elif(op.op.op_name == "Maxpool"):
				exptemp = prev[0]
				for i in range(1, nprev):
					cond = (True, 'Bool')
					for j in range(nprev):
						if i!=j:
							cond = AND(cond, GEQ(prev[i], prev[j]))
					exptemp = IF(cond, prev[i], exptemp)

				exptemp = s.os.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)
			
			elif(op.op.op_name == "rev_Maxpool"):
				total_list = curr_list + [(curr.name, "Neuron")]
				exptemp = total_list[0]
				for i in range(1, self.Ncurr+1):
					cond = (True, 'Bool')
					for j in range(self.Ncurr+1):
						if i!=j:
							cond = AND(cond, GEQ(total_list[i], total_list[j]))
					exptemp = IF(cond, total_list[i], exptemp)
				prevexp = (0, "Float") 
				for i in range(len(prev)):
					prevexp = ADD(prevexp, prev[i])

				exptemp = s.os.convertToZ3(exptemp)
				prevexp = s.os.convertToZ3(prevexp)
				s.currop = (prevexp == exptemp)
			
			for m in curr.symmap.keys():
				curr_prime.symmap[m] = curr.symmap[m]
			
			
			# computation = (curr_prime.name == curr.name)
			# computation = s.os.tempC
			computation = [s.currop] + s.os.tempC
			# print(computation)
			s.os.tempC = []
			s.flag = True
			s.visit(op.ret)
			print("graph expansion done", time.time())
			vallist = None
			#s.os.flag = True
			if(isinstance(op.ret, AST.TransRetIfNode)):
				vallist = self.visitTransRetIf(op.ret, s)
			else:
				#print('here')
				vallist = self.visitTransRetBasic(op.ret, s)
			print("symbolic output", time.time())
			# print(computation)
			# print(s.currop)
			# print()
			# print(s.os.C)
			# print()
			# print(s.os.C)
			leftC = computation + s.os.C 
			# print(s.os.C)
			
			# leftC = And(And(And(s.os.convertToZ3(s.os.C)), computation), s.currop)

			self.applyTrans(leftC, vallist, s, curr_prime, computation)
			print("Proved ", op.op.op_name)

	def applyTrans(self, leftC, vallist, s, curr_prime, computation):
		# print(leftC)
		if(isinstance(vallist, list)):
			# print(leftC)
			for (elem, val) in zip(self.shape.keys(), vallist):
				curr_prime.symmap[elem] = val
				# print(elem)
				# print(s.os.convertToZ3(val))
				# jshdfkj
		

			# print(self.store)
			# print(s.os.store)
			#Put this somewhere before so it only runs once
			conslist = [self.constraint]
			if isinstance(self.constraint, AST.ExprListNode):
				conslist = self.constraint.exprlist
			set_option(max_args=10000000, max_lines=1000000, max_depth=10000000, max_visited=1000000)
			for one_cons in conslist:
				c = populate_vars(s.vars, curr_prime, self.C, self.store, s.os, one_cons, self.number, False)
				# print('here')
				z3constraint = s.os.convertToZ3(c)
				z3constraint = substitute(z3constraint, (curr_prime.name, exptemp))
				#print("time", time.time())
				if len(self.E) > 0:
					conds_eps = [z3constraint]
					for e in self.E:
						conds_eps.append(e <= 1)
						conds_eps.append(e >= -1)
					z3constraint = Exists(self.E, And(conds_eps))
				#solver.set(timeout=30)
				# leftC += s.os.C 
				# print(leftC)
				newLeftC = leftC + s.os.tempC
				# print(computation)
				# print()
				# print("@@@@@@@@@@@@@@@@@@@@@@@@")
				# for i in newLeftC:
				# 	print(i)
				# print()
				# print("@@@@@@@@@@@@@@@@@@@@@@@@")
				# print(z3constraint)
				# print("@@@@@@@@@@@@@@@@@@@@@@@@")
				opt = Opt_solver()
				solver = Opt_solver()
				print("gen",time.time())
				lhs = And(newLeftC)
				rhs = z3constraint
				# print()
				# print(lhs)
				# print()
				# print(rhs)
				# print()
				# fc
				w = solver.solve(lhs, rhs)
				if(not w):
					print("end",time.time())
					#print(solver)
					#print(solver.model())
					raise Exception("Transformer"+ " " + " not true")
				print("end",time.time())
				s.os.tempC = []
				# print(newLeftC)
				# print(z3constraint)
				# print(w)
				# kjsdh
				# opt = optimization(z3constraint)
				# flag = True 
				# opt = None
				# if opt:
				# 	d = z3constraint.decl()
				# 	flag = True 
				# 	for i in range(len(opt)):
				# 		solver = Solver()
				# 		p = Not(Implies(And(newLeftC), d(opt[i][0], opt[i][1])))
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
				# if not flag:
				# 	solver = Solver()
				# 	p = Not(Implies(And(newLeftC), z3constraint))
				# 	#set_option(max_args=10000000, max_lines=1000000, max_depth=10000000, max_visited=1000000)
				# 	#print("final query")
				# 	#print(p)
				# 	#print("------------------------------------------------------------------------------")
				# 	solver.add(p)
				# 	print("gen",time.time())
				# 	#print(solver)
				# 	#Printing stats about Z3 Queries:
				# 	#print(len(get_vars(p)) )
				# 	if(not (solver.check() == unsat)):
				# 		print("end",time.time())
				# 		#print(solver)
				# 		#print(solver.model())
				# 		raise Exception("Transformer"+ " " + " not true")
				# 	print("end",time.time())
				# s.os.tempC = []
		else:
			condz3 = s.os.convertToZ3(vallist.cond)
			preC = leftC + s.os.tempC
			s.os.tempC = []
			self.applyTrans(preC + [condz3], vallist.left, s, curr_prime, computation)
			self.applyTrans(preC + [Not(condz3)], vallist.right, s, curr_prime, computation)

	
	def visitTransRetBasic(self, node, s):
		return s.os.visit(node.exprlist)

	def visitTransRetIf(self, node, s):
		cond = s.os.visit(node.cond)
		left = None
		right = None
		if(isinstance(node.tret, AST.TransRetIfNode)):
			left = self.visitTransRetIf(node.tret, s)
		else:
			left = self.visitTransRetBasic(node.tret, s)

		if(isinstance(node.fret, AST.TransRetIfNode)):
			right = self.visitTransRetIf(node.fret, s)
		else:
			right = self.visitTransRetBasic(node.fret, s)

		return IF(cond, left, right)

	def visitSeq(self, node: AST.SeqNode):
		self.visit(node.stmt1)
		self.visit(node.stmt2)

	def visitProg(self, node):
		self.visit(node.shape)
		self.visit(node.stmt)


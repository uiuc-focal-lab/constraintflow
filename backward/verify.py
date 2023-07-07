import astVisitor
import astcf as AST
from z3 import *
#from cvc5.pythonic import * 
from value import *
from symbolicos import *
from symbolicgraph import *
import time
set_param('parallel.enable', True) #uncomment when using Z3

class AstRefKey:
    def __init__(self, n):
        self.n = n
    def __hash__(self):
        return self.n.hash()
    def __eq__(self, other):
        return self.n.eq(other.n)
    def __repr__(self):
        return str(self.n)

def askey(n):
    assert isinstance(n, AstRef)
    return AstRefKey(n)

def get_vars(f):
    r = set()
    def collect(f):
      if is_const(f): 
          if f.decl().kind() == Z3_OP_UNINTERPRETED and not askey(f) in r:
              r.add(askey(f))
      else:
          for c in f.children():
              collect(c)
    collect(f)
    return r

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
		print("start",time.time())
		node = self.theta[node.trans.name]
		for op_i in range(len(node.oplist.olist)):
			self.E.clear()

			op = node.oplist.olist[op_i]
			store = self.store
			if(op.op.op_name == "Relu"):
				nprev= 1
			else:
				nprev = self.Nprev
			s = SymbolicGraph(self.store, self.F, self.constraint, self.shape, nprev, self.Nzono, self.number, self.M, self.V, self.C, self.E, self.old_eps, self.old_neurons)
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
				if(not "bias" in curr.symmap.keys()):
					curr.symmap["bias"] = ((Real('bias_curr' + str(self.number.nextn())), "Float"))
				if(not "weight" in curr.symmap.keys()):
					curr.symmap["weight"] = [(Real('weight_curr' + str(op_i) + "_" + str(self.number.nextn())), "Float") for i in range(nprev)]
				exptemp = curr.symmap["bias"]
				for i in range(len(prev)):
					exptemp = ADD(exptemp, MULT(curr.symmap["weight"][i], prev[i]))

				exptemp = s.os.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)
			elif(op.op.op_name == "rev_Affine"):
				if(not "equations" in curr.symmap.keys()):
					curr.symmap["equations"] = [(Real('equations_' + str(op_i) + "_" + str(self.number.nextn())), "PolyExp") for i in range(self.Ncurr)]
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
			computation = [s.currop, (curr_prime.name == curr.name)]

			s.visit(op.ret)
			vallist = None
			if(isinstance(op.ret, AST.TransRetIfNode)):
				vallist = self.visitTransRetIf(op.ret, s)
			else:
				vallist = self.visitTransRetBasic(op.ret, s)
				
			# print(computation)
			# print(s.currop)
			# print()
			# print(s.os.C)
			# print()
			# print(s.os.C)
			leftC = s.os.C + computation
			# print(vallist)
			# leftC = And(And(And(s.os.convertToZ3(s.os.C)), computation), s.currop)

			self.applyTrans([], vallist, s, curr_prime, computation)
			print("Proved ", op.op.op_name)

	def applyTrans(self, leftC, vallist, s, curr_prime, computation):
		# print(vallist)
		# print(leftC)
		if(isinstance(vallist, list)):
			# print(leftC)
			for (elem, val) in zip(self.shape.keys(), vallist):
				curr_prime.symmap[elem] = val

			# print(self.store)
			# print(s.os.store)
			#Put this somewhere before so it only runs once
			conslist = [self.constraint]
			if isinstance(self.constraint, AST.ExprListNode):
				conslist = self.constraint.exprlist

			for one_cons in conslist:
				c = populate_vars(s.vars, curr_prime, self.C, self.store, s.os, one_cons, self.number, False)
			# print(c)
			#for c in clist:
				z3constraint = s.os.convertToZ3(c)
				if len(self.E) > 0:
					conds_eps = [z3constraint]
					for e in self.E:
						conds_eps.append(e <= 1)
						conds_eps.append(e >= -1)
					z3constraint = Exists(self.E, And(conds_eps))
				solver = Solver()
				#solver.set(timeout=30)
				leftC += s.os.C + computation
				# print(leftC)

				p = Not(Implies(And(leftC), z3constraint))
				#print(p)
				#print("------------------------------------------------------------------------------")
				solver.add(p)
				print("gen",time.time())
				
				#Printing stats about Z3 Queries:
				#print(len(get_vars(p)) )
				if(not (solver.check() == unsat)):
					print("end",time.time())
					raise Exception("Transformer"+ " " + " not true")
				print("end",time.time())
		else:
			condz3 = s.os.convertToZ3(vallist.cond)
			self.applyTrans(leftC + [condz3], vallist.left, s, curr_prime, computation)
			self.applyTrans(leftC + [Not(condz3)], vallist.right, s, curr_prime, computation)

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

	# def visitFlow(self, node):
	# 	pass

	def visitSeq(self, node: AST.SeqNode):
		self.visit(node.stmt1)
		self.visit(node.stmt2)

	def visitProg(self, node):
		self.visit(node.shape)
		self.visit(node.stmt)

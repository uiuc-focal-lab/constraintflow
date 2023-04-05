import astVisitor
import ast as AST
from z3 import *
from value import *
from symbolicos import *
from symbolicgraph import *


class Verify(astVisitor.ASTVisitor):

	def __init__(self):
		self.shape = {}
		self.F = {}
		self.theta = {}
		self.N = 3
		self.number = Number()
		self.M = {}
		self.V = {}
		self.C = []
		self.store = {}

	def visitShapeDecl(self, node):
		for (t,e) in node.elements.arglist:
			self.shape[e.name] = t.name

		self.constraint = node.p

	def visitFunc(self, node):
		self.F[node.decl.name.name] = node

	def visitTransformer(self, node):
		self.theta[node.name.name] = node

	def visitFlow(self, node):
		store = self.store
		s = SymbolicGraph(self.store, self.F, self.constraint, self.shape, self.N, self.number, self.M, self.V, self.C)
		node = self.theta[node.trans.name]
		curr = Vertex('Curr')
		self.V[curr.name] =  curr 
		populate_vars(s.vars, curr, self.C, self.store, s.os, self.constraint, self.number)
		prev = []
		for i in range(self.N):
			p = Vertex('Prev'+str(i))
			prev.append((p.name, "Neuron"))
			self.V[p.name] = p
			populate_vars(s.vars, p, self.C, self.store, s.os, self.constraint, self.number)

		store["curr"] = (curr.name, "Neuron")
		store["prev"] = prev 

		for op_i in range(len(node.oplist.olist)):
			op = node.oplist.olist[op_i]
			curr_prime = Vertex('curr_prime' + str(op_i))
			s.V[curr_prime.name] = curr_prime

			#Define relationship between curr and prev
			if(op.op.op_name == "Affine"):
				if(not "bias" in curr.symmap.keys()):
					curr.symmap["bias"] = ((Real('bias_curr' + str(self.number.nextn())), "Float"))
				if(not "weight" in curr.symmap.keys()):
					curr.symmap["weight"] = [(Real('weight_curr' + str(op_i) + "_" + str(self.number.nextn())), "Float") for i in range(self.N)]
				exptemp = curr.symmap["bias"]
				for i in range(len(prev)):
					exptemp = ADD(exptemp, MULT(curr.symmap["weight"][i], prev[i]))
					

			elif(op.op.op_name == "Relu"):
				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = ADD(exptemp, prev[i])
				exptemp = IF(GEQ(exptemp, (0, 'Float')), exptemp, (0, 'Float'))

			else: #Maxpool
				exptemp = prev[0]
				for i in range(1, self.N):
					cond = (True, 'Bool')
					for j in range(self.N):
						if i!=j:
							cond = AND(cond, GEQ(prev[i], prev[j]))
					exptemp = IF(cond, prev[i], exptemp)
			
			for m in curr.symmap.keys():
				curr_prime.symmap[m] = curr.symmap[m]
			
			exptemp = s.os.convertToZ3(exptemp)
			s.currop = (curr.name == exptemp) #Would it be safe to just add this to s.C?
			computation = (curr_prime.name == curr.name)

			s.visit(op.ret)
			vallist = None
			if(isinstance(op.ret, AST.TransRetIfNode)):
				vallist = self.visitTransRetIf(op.ret, s)
			else:
				vallist = self.visitTransRetBasic(op.ret, s)
				
			# print(computation)
			# print(s.currop)
			# print(s.os.C)
			leftC = And(And(And(s.os.convertToZ3(s.os.C)), computation), s.currop)

			self.applyTrans(leftC, vallist, s, curr_prime)
			print("Proved ", op.op.op_name)

	def applyTrans(self, leftC, vallist, s, curr_prime):
		if(isinstance(vallist, list)):
			# print(leftC)
			for (elem, val) in zip(self.shape.keys(), vallist):
				curr_prime.symmap[elem] = val

			# print(self.store)
			# print(s.os.store)
			c = populate_vars(s.vars, curr_prime, self.C, self.store, s.os, self.constraint, self.number, False)
			z3constraint = s.os.convertToZ3(c)
			solver = Solver()
			p = Not(Implies(leftC, z3constraint))
			# print(p)
			solver.add(p)
			if(not (solver.check() == unsat)):
				raise Exception("Transformer"+ " " + " not true")
		else:
			condz3 = s.os.convertToZ3(vallist.cond)
			self.applyTrans(And(leftC, condz3), vallist.left, s, curr_prime)
			self.applyTrans(And(leftC, Not(condz3)), vallist.right, s, curr_prime)

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

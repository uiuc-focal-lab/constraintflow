import astVisitor
import ast as AST
from z3 import *
from value import *
from verify import *

class CreateF:

	def __init__(self):
		self.shape = {}
		self.F = {}
		self.theta = {}
		self.N = 3

	def visitShapeDecl(self, node):
		for (t,e) in node.elements.arglist.arglist:
			self.shape[e.name] = t

		self.constraint = node.p

	def visitFunc(self, node):
		self.F[f.name.name] = f

	def visitTransformer(self, node):
		self.theta[node.name] = node
		store = {}
		curr = Vertex('Curr')
		V = {}
		V[curr.name] = curr
		prev = []
		for i in range(self.N):
			p = Vertex('Prev'+str(i))
			prev.append((p.name, "Neuron"))
			V[p.name] = p

		store["curr"] = (curr.name, "Neuron")
		store["prev"] = prev
		s = SymbolicGraph(store, F)
		s.V = V

		for i in len(node.oplist):
			op = node.oplist[i]
			s.visit(op.ret)
			vallist = s.os.visit(op.ret)
			currprime = Vertex('Currprime' + str(i))
			V[currprime.name] = currprime
			store["curr'"] = (currprime.name, "Neuron")
			for (elem, val) in zip(shape.keys(), vallist):
				currprime.symmap[elem] = val

			z3constraint = s.os.visit(self.constraint)
			s = Solver()
			p = Not(Implies(s.os.C, z3constraint))
			if(not (s.check() == unsat)):
				raise Exception("Transformer" + str(node.name.name) + " " + str(op.op) + " not true")


	def visitFlow(self, node):
		pass

	def visitSeq(self, node: AST.SeqNode):
		self.visit(node.stmt1)
		self.visit(node.stmt2)

	def visitProg():
		self.visit(node.shape)
		self.visit(node.stmt)

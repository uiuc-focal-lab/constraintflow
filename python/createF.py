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
		s.visit()
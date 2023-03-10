import astVisitor
import ast as AST
from z3 import *
from value import *
from verify import *


class SymbolicGraph(astVisitor):

	def __init__(self, store, F):
		self.M = {}
		self.V = {}
		self.C = []
		self.store = store
		self.F = F
		self.shape = []
		self.os = Evaluate(self.store, self.F, self.M, self.V, self.C, self.shape)
		self.number = Number()
		self.N = 3

	def vistInt(self, node):
		pass

	def visitVar(self, node):
		pass

	def visitFloat(self, node):
		pass

	def visitBool(self, node):
		pass

	def visitBinop(self, node):
		self.visit(left)
		self.visit(right)

	def visitUnOp(self, node):
		self.visit(node.expr)

	def visitSum(self, node):
		self.visit(node.expr)

	def visitDot(self, node):
		self.visit(left)
		self.visit(right)

	def visitGetMetadata(self, node):
		self.visit(node.expr)
		n = self.os.visit(node.expr)
		if not node.metadata in self.V[n].symmap.keys():
			newvar = None
			if(node.metadata == "bias"):
				newvar = (Real('X' + self.number.nextn()), "Bool")
			elif(node.metadate == "weight"):
				newvar = [(Real('X' + self.number.nextn()), "Float") for i in self.N]
			elif(node.metadata == "layer"):
				newvar = (Int('X' + self.number.nextn()), "Int")
			self.V[n].symmap[node.metadata] = newvar

	def visitGetElement(self, node):
		self.visit(node.expr)
		n = self.os.visit(node.expr)
		if not node.elem in self.V[n].symmap.keys():
			newvar = None
			if(shape[node.elem] == "Bool"):
				newvar = (Bool('X' + self.number.nextn()), "Bool")
			elif(shape[node.elem] == "Int"):
				newvar = (Int('X' + self.number.nextn()), "Int")
			else:
				newvar = (Real('X' + self.number.nextn()), "Float")
			self.V[n].symmap[node.elem] = newvar

	def visitMap(self, node):
		self.visit(node.expr)
		p = self.os.visit(node.expr)
		if(isinstance(p, tuple)):


class checkPoly(astVisitor):

	def __init__(self, os):
		self.os = os

	#For all other ones, recursively call the children/pass

	def visitGetElement(self, node):
		n = self.os.visit(node.expr)
		if(isinstance(self.os.V[n[0]].symmap[n.elem], tuple):
			if(self.os.V[n[0]].symmap[n.elem][1] == "Float"):
				oldvar = self.os.V[n[0]].symmap[n.elem][0]
				newvar = (Real('X' + self.number.nextn()), "Float")

				for i in range(self.N):
					neuron = Vertex('X' + self.number.nextn())
					V[neuron.name] = neuron 
					const = (Real('X' + self.number.nextn()), "Float")
					newvar = Add(newvar, Mult(const, (neuron, "Neuron")))

				self.os.V[n[0]].symmap[n.elem] = newvar
				self.os.C.append(oldvar == self.os.convertToZ3(newvar))
